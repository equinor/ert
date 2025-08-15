from abc import ABC
from typing import Annotated, Any, cast

import numpy as np
import polars as pl
from pydantic import Field, PrivateAttr

from ert.config import (
    DesignMatrix,
    EverestConstraintsConfig,
    EverestObjectivesConfig,
    ExtParamConfig,
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config import Field as FieldConfig
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_arg import create_run_arguments
from ert.run_models.run_model import RunModel
from ert.storage.local_ensemble import LocalEnsemble


class InitialEnsembleRunModel(RunModel, ABC):
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[
        Annotated[
            (GenKwConfig | SurfaceConfig | FieldConfig | ExtParamConfig),
            Field(discriminator="type"),
        ]
    ]
    response_configuration: list[
        Annotated[
            (
                GenDataConfig
                | SummaryConfig
                | EverestConstraintsConfig
                | EverestObjectivesConfig
            ),
            Field(discriminator="type"),
        ]
    ]
    ert_templates: list[tuple[str, str]]
    _observations: dict[str, pl.DataFrame] | None = PrivateAttr()

    def __init__(
        self, *, observations: dict[str, pl.DataFrame] | None, **data: Any
    ) -> None:
        super().__init__(**data)
        self._observations = observations

    def _sample_and_evaluate_ensemble(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        simulation_arguments: dict[str, str] | None,
        ensemble_name: str,
        rerun_failed_realizations: bool = False,
        ensemble_storage: LocalEnsemble | None = None,
    ) -> LocalEnsemble:
        parameters_config, design_matrix, design_matrix_group = (
            self._merge_parameters_from_design_matrix(
                cast(list[ParameterConfig], self.parameter_configuration),
                self.design_matrix,
                rerun_failed_realizations,
            )
        )
        if ensemble_storage is None:
            experiment_storage = self._storage.create_experiment(
                parameters=parameters_config
                + ([design_matrix_group] if design_matrix_group else []),
                observations=self._observations,
                responses=cast(list[ResponseConfig], self.response_configuration),
                simulation_arguments=simulation_arguments,
                name=self.experiment_name,
                templates=self.ert_templates,
            )
            ensemble_storage = self._storage.create_ensemble(
                experiment_storage,
                ensemble_size=self.ensemble_size,
                name=ensemble_name,
            )
            if design_matrix_group is not None and design_matrix is not None:
                save_design_matrix_to_ensemble(
                    design_matrix.design_matrix_df,
                    ensemble_storage,
                    np.where(self.active_realizations)[0],
                    design_matrix_group.name,
                )
            if hasattr(self, "_ensemble_id"):
                setattr(self, "_ensemble_id", ensemble_storage.id)  # noqa: B010

        assert ensemble_storage is not None
        self.set_env_key("_ERT_EXPERIMENT_ID", str(ensemble_storage.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble_storage.id))

        sample_prior(
            ensemble_storage,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in parameters_config],
            random_seed=self.random_seed,
        )

        prior_args = create_run_arguments(
            self._run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=ensemble_storage,
        )
        self._evaluate_and_postprocess(
            prior_args,
            ensemble_storage,
            evaluator_server_config,
        )
        return ensemble_storage
