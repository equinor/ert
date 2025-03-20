from __future__ import annotations

import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

import numpy as np

from ert.config import ErtConfig, ESSettings, HookRuntime, UpdateSettings
from ert.config.parsing.config_errors import ConfigValidationError
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Storage
from ert.trace import tracer

from ..run_arg import create_run_arguments
from .base_run_model import ErtRunError, StatusEvents, UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class EnsembleSmoother(UpdateRunModel):
    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        random_seed: int | None,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: UpdateSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
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
            start_iteration=0,
            total_iterations=2,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
        )
        self.target_ensemble_format = target_ensemble
        self.experiment_name = experiment_name

        self.support_restart = False

        self._parameter_configuration = config.ensemble_config.parameter_configuration
        self._design_matrix = config.analysis_config.design_matrix
        self._observations = config.observations
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
        self.run_workflows(
            HookRuntime.PRE_EXPERIMENT,
            fixtures={"random_seed": self.random_seed},
        )
        ensemble_format = self.target_ensemble_format
        experiment = self._storage.create_experiment(
            parameters=parameters_config
            + ([design_matrix_group] if design_matrix_group else []),
            observations=self._observations,
            responses=self._response_configuration,
            name=self.experiment_name,
        )

        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        prior = self._storage.create_ensemble(
            experiment,
            ensemble_size=self.ensemble_size,
            name=ensemble_format % 0,
        )
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
        posterior = self.update(prior, ensemble_format % 1)

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
        self.run_workflows(
            HookRuntime.POST_EXPERIMENT,
            fixtures={
                "random_seed": self.random_seed,
                "storage": self._storage,
                "ensemble": posterior,
            },
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"
