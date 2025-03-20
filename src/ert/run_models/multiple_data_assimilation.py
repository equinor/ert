from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.config import (
    ConfigValidationError,
    ErtConfig,
    ESSettings,
    HookRuntime,
    UpdateSettings,
)
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble, Storage
from ert.trace import tracer

from ..run_arg import create_run_arguments
from .base_run_model import ErtRunError, StatusEvents, UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig

logger = logging.getLogger(__name__)

MULTIPLE_DATA_ASSIMILATION_GROUP = "Parameter update"


class MultipleDataAssimilation(UpdateRunModel):
    """
    Run multiple data assimilation (MDA) ensemble smoother with custom weights.
    """

    default_weights = "4, 2, 1"

    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str | None,
        restart_run: bool,
        prior_ensemble_id: str,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        random_seed: int | None,
        weights: str,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        self._relative_weights = weights
        self._parameter_configuration = config.ensemble_config.parameter_configuration
        self._design_matrix = config.analysis_config.design_matrix
        self.weights = self.parse_weights(weights)

        self.target_ensemble_format = target_ensemble
        self.experiment_name = experiment_name
        self.restart_run = restart_run
        self.prior_ensemble_id = prior_ensemble_id
        start_iteration = 0
        total_iterations = len(self.weights) + 1
        if self.restart_run:
            if not self.prior_ensemble_id:
                raise ValueError("For restart run, prior ensemble must be set")
            start_iteration = storage.get_ensemble(prior_ensemble_id).iteration + 1
            total_iterations -= start_iteration
        elif not self.experiment_name:
            raise ValueError("For non-restart run, experiment name must be set")
        super().__init__(
            es_settings,
            update_settings,
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.model_config,
            queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.ert_templates,
            config.hooked_workflows,
            active_realizations=active_realizations,
            total_iterations=total_iterations,
            start_iteration=start_iteration,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
        )
        self.support_restart = False
        self._observations = config.observations
        self._parameter_configuration = config.ensemble_config.parameter_configuration
        self._response_configuration = config.ensemble_config.response_configuration

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()

        parameters_config = self._parameter_configuration
        design_matrix = self._design_matrix
        design_matrix_group = None
        if design_matrix is not None:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )
            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc

        self.restart = restart
        if self.restart_run:
            id = self.prior_ensemble_id
            try:
                ensemble_id = UUID(id)
                prior = self._storage.get_ensemble(ensemble_id)
                experiment = prior.experiment
                self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
                self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
                assert isinstance(prior, Ensemble)
                if self.start_iteration != prior.iteration + 1:
                    raise ValueError(
                        f"Experiment misconfigured, got starting iteration: {self.start_iteration},"
                        f"restart iteration = {prior.iteration + 1}"
                    )
            except (KeyError, ValueError) as err:
                raise ErtRunError(
                    f"Prior ensemble with ID: {id} does not exists"
                ) from err
        else:
            self.run_workflows(
                HookRuntime.PRE_EXPERIMENT,
                fixtures={"random_seed": self.random_seed},
            )
            sim_args = {"weights": self._relative_weights}
            experiment = self._storage.create_experiment(
                parameters=parameters_config
                + ([design_matrix_group] if design_matrix_group else []),
                observations=self._observations,
                responses=self._response_configuration,
                simulation_arguments=sim_args,
                name=self.experiment_name,
            )

            prior = self._storage.create_ensemble(
                experiment,
                ensemble_size=self.ensemble_size,
                iteration=0,
                name=self.target_ensemble_format % 0,
            )
            self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
            self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
            prior_args = create_run_arguments(
                self.run_paths,
                np.array(self.active_realizations, dtype=bool),
                ensemble=prior,
            )

            sample_prior(
                prior,
                np.where(self.active_realizations)[0],
                parameters=[param.name for param in parameters_config],
                random_seed=self.random_seed,
            )

            if design_matrix_group is not None and design_matrix is not None:
                save_design_matrix_to_ensemble(
                    design_matrix.design_matrix_df,
                    prior,
                    np.where(self.active_realizations)[0],
                    design_matrix_group.name,
                )
            self._evaluate_and_postprocess(
                prior_args,
                prior,
                evaluator_server_config,
            )
        enumerated_weights = list(enumerate(self.weights))
        weights_to_run = enumerated_weights[prior.iteration :]

        for iteration, weight in weights_to_run:
            posterior = self.update(
                prior,
                self.target_ensemble_format % (iteration + 1),
                weight=weight,
            )
            posterior_args = create_run_arguments(
                self.run_paths,
                self.active_realizations,
                ensemble=posterior,
            )
            self._evaluate_and_postprocess(
                posterior_args,
                posterior,
                evaluator_server_config,
            )
            prior = posterior

        self.run_workflows(
            HookRuntime.POST_EXPERIMENT,
            fixtures={
                "random_seed": self.random_seed,
                "storage": self._storage,
                "ensemble": prior,
            },
        )

    @staticmethod
    def parse_weights(weights: str) -> list[float]:
        """Parse weights string and scale weights such that their reciprocals sum
        to 1.0, i.e., sum(1.0 / x for x in weights) == 1.0. See for example Equation
        38 of evensen2018 - Analysis of iterative ensemble
        smoothers for solving inverse problems.
        """
        if not weights:
            raise ValueError(f"Must provide weights, got {weights}")

        elements = weights.split(",")
        elements = [element.strip() for element in elements if element.strip()]

        result: list[float] = []
        for element in elements:
            try:
                f = float(element)
                if f == 0:
                    logger.info("Warning: 0 weight, will ignore")
                else:
                    result.append(f)
            except ValueError as e:
                raise ValueError(f"Warning: cannot parse weight {element}") from e
        if not result:
            raise ValueError(f"Invalid weights: {weights}")

        length = sum(1.0 / x for x in result)
        return [x * length for x in result]

    @classmethod
    def name(cls) -> str:
        return "Multiple data assimilation"

    @classmethod
    def display_name(cls) -> str:
        return cls.name() + " - Recommended algorithm"

    @classmethod
    def description(cls) -> str:
        return "[Sample|restart] → [evaluate → update] for each weight."

    @classmethod
    def group(cls) -> str | None:
        return MULTIPLE_DATA_ASSIMILATION_GROUP
