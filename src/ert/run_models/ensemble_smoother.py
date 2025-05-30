from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from pydantic import PrivateAttr

from ert.config.parsing.config_errors import ConfigValidationError
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.trace import tracer

from ..config import DesignMatrix, ParameterConfig, ResponseConfig
from ..plugins import PostExperimentFixtures, PreExperimentFixtures
from ..run_arg import create_run_arguments
from .base_run_model import ErtRunError, UpdateRunModel

logger = logging.getLogger(__name__)


class EnsembleSmoother(UpdateRunModel):
    target_ensemble: str
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[ParameterConfig]
    response_configuration: list[ResponseConfig]
    ert_templates: list[tuple[str, str]]

    start_iteration: int = 0
    _total_iterations: int = PrivateAttr(default=2)

    _observations: dict[str, pl.DataFrame] = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        observations = data.pop("observations", None)
        super().__init__(**data)
        self._observations = observations

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        parameters_config = self.parameter_configuration
        design_matrix = self.design_matrix
        design_matrix_group = None
        if design_matrix is not None and not restart:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )
                if not any(p.update for p in parameters_config):
                    raise ConfigValidationError(
                        "No parameters to update as all "
                        "parameters were set to update:false!",
                    )
            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc

        self._restart = restart
        self.run_workflows(
            fixtures=PreExperimentFixtures(random_seed=self.random_seed),
        )
        ensemble_format = self.target_ensemble
        experiment = self._storage.create_experiment(
            parameters=parameters_config
            + ([design_matrix_group] if design_matrix_group else []),
            observations=self._observations,
            responses=self.response_configuration,
            name=self.experiment_name,
            templates=self.ert_templates,
        )

        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        prior = self._storage.create_ensemble(
            experiment,
            ensemble_size=self.ensemble_size,
            name=ensemble_format % 0,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_args = create_run_arguments(
            self._run_paths,
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
            self._run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=posterior,
        )

        self._evaluate_and_postprocess(
            posterior_args,
            posterior,
            evaluator_server_config,
        )
        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=posterior,
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → update → evaluate"
