from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any

import numpy as np
import polars as pl
from pydantic import PrivateAttr

from ert.config import DesignMatrix, HookRuntime, ParameterConfig, ResponseConfig
from ert.config.parsing.config_errors import ConfigValidationError
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble
from ert.trace import tracer

from ..analysis import ErtAnalysisError, enif_update
from ..plugins import (
    PostExperimentFixtures,
    PostUpdateFixtures,
    PreExperimentFixtures,
    PreFirstUpdateFixtures,
    PreUpdateFixtures,
)
from ..run_arg import create_run_arguments
from .base_run_model import ErtRunError, UpdateRunModel
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent

logger = logging.getLogger(__name__)


class EnsembleInformationFilter(UpdateRunModel):
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[ParameterConfig]
    response_configuration: list[ResponseConfig]
    ert_templates: list[tuple[str, str]]

    start_iteration: int = 0

    _observations: dict[str, pl.DataFrame] = PrivateAttr()
    _total_iterations: int = PrivateAttr(default=2)

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
        if self.design_matrix is not None:
            try:
                parameters_config = self.design_matrix.merge_with_existing_parameters(
                    parameters_config
                )
            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc

        self._restart = restart
        self.run_workflows(
            PreExperimentFixtures(random_seed=self.random_seed),
        )
        ensemble_format = self.target_ensemble
        experiment = self._storage.create_experiment(
            parameters=parameters_config,
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
            design_matrix_df=self.design_matrix.design_matrix_df
            if self.design_matrix is not None
            else None,
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

    def update(
        self,
        prior: Ensemble,
        posterior_name: str,
        weight: float = 1.0,
    ) -> Ensemble:
        self.validate_successful_realizations_count()
        self.send_event(
            RunModelUpdateBeginEvent(iteration=prior.iteration, run_id=prior.id)
        )
        self.send_event(
            RunModelStatusEvent(
                iteration=prior.iteration,
                run_id=prior.id,
                msg="Creating posterior ensemble..",
            )
        )

        pre_first_update_fixtures = PreFirstUpdateFixtures(
            storage=self._storage,
            ensemble=prior,
            observation_settings=self.update_settings,
            es_settings=self.analysis_settings,
            random_seed=self.random_seed,
            reports_dir=self.reports_dir(experiment_name=prior.experiment.name),
            run_paths=self._run_paths,
        )

        posterior = self._storage.create_ensemble(
            prior.experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=posterior_name,
            prior_ensemble=prior,
        )
        if prior.iteration == 0:
            self.run_workflows(
                fixtures=pre_first_update_fixtures,
            )

        update_args_dict = {
            field.name: getattr(pre_first_update_fixtures, field.name)
            for field in dataclasses.fields(pre_first_update_fixtures)
        }

        self.run_workflows(
            fixtures=PreUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.PRE_UPDATE}
            ),
        )
        try:
            enif_update(
                prior,
                posterior,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observation_keys,
                global_scaling=weight,
                random_seed=self.random_seed,
                progress_callback=functools.partial(
                    self.send_smoother_event,
                    prior.iteration,
                    prior.id,
                ),
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Update algorithm failed for iteration:"
                f"{posterior.iteration}. The following error occurred: {e}"
            ) from e

        self.run_workflows(
            fixtures=PostUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.POST_UPDATE}
            ),
        )
        return posterior

    @classmethod
    def name(cls) -> str:
        return "Ensemble Information Filter (Experimental)"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → EnIF update → evaluate"
