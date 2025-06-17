from __future__ import annotations

import logging
from typing import Any, ClassVar
from uuid import UUID

import numpy as np
import polars as pl
from pydantic import PrivateAttr

from ert.config import (
    ConfigValidationError,
    DesignMatrix,
    ParameterConfig,
    ResponseConfig,
)
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble, Experiment
from ert.trace import tracer

from ..plugins import PostExperimentFixtures, PreExperimentFixtures
from ..run_arg import create_run_arguments
from .base_run_model import BaseRunModel, ErtRunError

logger = logging.getLogger(__name__)


class EnsembleExperiment(BaseRunModel):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration.<br>It will never overwrite existing ensembles, and
    will always sample parameters.<br>
    """

    ensemble_name: str
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[ParameterConfig]
    response_configuration: list[ResponseConfig]
    ert_templates: list[tuple[str, str]]
    _observations: dict[str, pl.DataFrame] = PrivateAttr()
    _experiment_id: UUID | None = PrivateAttr(None)
    _ensemble_id: UUID | None = PrivateAttr(None)
    supports_rerunning_failed_realizations: ClassVar[bool] = True

    def __init__(self, **data: Any) -> None:
        observations = data.pop("observations", None)
        super().__init__(**data)
        self._observations = observations

    @property
    def _ensemble(self) -> Ensemble:
        assert self._ensemble_id is not None
        return self._storage.get_ensemble(self._ensemble_id)

    @property
    def _experiment(self) -> Experiment:
        assert self._experiment_id is not None
        return self._storage.get_experiment(self._experiment_id)

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        self._is_rerunning_failed_realizations = rerun_failed_realizations
        # If design matrix is present, we try to merge design matrix parameters
        # to the experiment parameters and set new active realizations
        parameters_config = self.parameter_configuration
        design_matrix = self.design_matrix
        design_matrix_group = None
        if design_matrix is not None and not rerun_failed_realizations:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )
            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc

        if not rerun_failed_realizations:
            self.run_workflows(
                fixtures=PreExperimentFixtures(random_seed=self.random_seed),
            )
            self._experiment_id = self._storage.create_experiment(
                name=self.experiment_name,
                parameters=(
                    [*parameters_config, design_matrix_group]
                    if design_matrix_group is not None
                    else parameters_config
                ),
                observations=self._observations,
                responses=self.response_configuration,
                templates=self.ert_templates,
            ).id
            self._ensemble_id = self._storage.create_ensemble(
                self._experiment,
                name=self.ensemble_name,
                ensemble_size=self.ensemble_size,
            ).id

        assert self._experiment
        assert self._ensemble

        self.set_env_key("_ERT_EXPERIMENT_ID", str(self._experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self._ensemble.id))

        run_args = create_run_arguments(
            self._run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=self._ensemble,
        )

        sample_prior(
            self._ensemble,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in parameters_config],
            random_seed=self.random_seed,
        )

        if design_matrix_group is not None and design_matrix is not None:
            save_design_matrix_to_ensemble(
                design_matrix.design_matrix_df,
                self._ensemble,
                np.where(self.active_realizations)[0],
                design_matrix_group.name,
            )

        self._evaluate_and_postprocess(
            run_args,
            self._ensemble,
            evaluator_server_config,
        )
        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=self._ensemble,
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate all realizations"

    @classmethod
    def group(cls) -> str | None:
        return None
