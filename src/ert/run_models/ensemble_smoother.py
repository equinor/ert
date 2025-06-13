from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from pydantic import PrivateAttr

from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.trace import tracer

from ..config import ResponseConfig
from ..plugins import PostExperimentFixtures, PreExperimentFixtures
from ..run_arg import create_run_arguments
from .base_run_model import HasDesignParameters, UpdateRunModel

logger = logging.getLogger(__name__)


class EnsembleSmoother(HasDesignParameters, UpdateRunModel):
    experiment_name: str
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
        parameters, design_parameters = self.experiment_parameters(
            include_design_matrix=not restart, require_updated_parameters=True
        )

        self._restart = restart
        self.run_workflows(
            fixtures=PreExperimentFixtures(random_seed=self.random_seed),
        )
        ensemble_format = self.target_ensemble
        experiment = self._storage.create_experiment(
            parameters=parameters + design_parameters,
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
            parameters=[p.name for p in parameters],
            random_seed=self.random_seed,
        )

        if not restart:
            self.save_design_parameters(
                target_ensemble=prior,
                active_realizations=np.where(self.active_realizations)[0],
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
