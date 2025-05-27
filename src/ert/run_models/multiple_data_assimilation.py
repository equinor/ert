from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.config import ErtConfig, ESSettings, ObservationSettings
from ert.run_arg import create_run_arguments
from ert.storage import Ensemble, Storage
from ert.storage.local_ensemble import LocalEnsemble

from .base_run_model import ErtRunError, StatusEvents
from .update_run_model import UpdateRunModel

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
        random_seed: int,
        weights: str,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: ObservationSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        start_iteration = 0
        total_iterations = len(self.parse_weights(weights)) + 1
        if restart_run:
            if not prior_ensemble_id:
                raise ValueError("For restart run, prior ensemble must be set")
            start_iteration = storage.get_ensemble(prior_ensemble_id).iteration + 1
            total_iterations -= start_iteration
        elif not experiment_name:
            raise ValueError("For non-restart run, experiment name must be set")
        super().__init__(
            es_settings,
            update_settings,
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.runpath_config,
            queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.hooked_workflows,
            active_realizations=active_realizations,
            total_iterations=total_iterations,
            start_iteration=start_iteration,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
            config=config,
            target_ensemble=target_ensemble,
            experiment_name=experiment_name,
        )
        self.restart_run = restart_run
        self.prior_ensemble_id = prior_ensemble_id
        self._relative_weights = weights
        self.weights = self.parse_weights(weights)
        self.simulation_arguments: dict[str, str] | None = {
            "weights": self._relative_weights
        }

    def _update_then_run_ensembles(self, evaluator_server_config, prior):
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
                np.array(self.active_realizations, dtype=bool),
                ensemble=posterior,
            )
            self._evaluate_and_postprocess(
                posterior_args,
                posterior,
                evaluator_server_config,
            )
            prior = posterior

        return prior

    @staticmethod
    def parse_weights(weights: str) -> list[float]:
        """Parse weights string and scale weights such that their reciprocals sum
        to 1.0, i.e., sum(1.0 / x for x in weights) == 1.0. See, for example, Equation
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

    def _preExperimentFixtures(self):
        if not self.restart_run:
            super()._preExperimentFixtures()

    def _evaluate_prior(
        self,
        design_matrix,
        design_matrix_group,
        evaluator_server_config,
        parameters_config,
    ) -> LocalEnsemble:
        if self.restart_run:
            id_ = self.prior_ensemble_id
            try:
                ensemble_id = UUID(id_)
                prior = self._storage.get_ensemble(ensemble_id)
                experiment = prior.experiment
                self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
                self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
                assert isinstance(prior, Ensemble)
                if self.start_iteration != prior.iteration + 1:
                    raise ValueError(
                        "Experiment misconfigured, got starting "
                        f"iteration: {self.start_iteration},"
                        f"restart iteration = {prior.iteration + 1}"
                    )
                else:
                    return prior
            except (KeyError, ValueError) as err:
                raise ErtRunError(
                    f"Prior ensemble with ID: {id_} does not exists"
                ) from err
        else:
            return super()._evaluate_prior(
                design_matrix,
                design_matrix_group,
                evaluator_server_config,
                parameters_config,
            )

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
