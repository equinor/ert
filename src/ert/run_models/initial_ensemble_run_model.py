from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field

from ert.config import (
    EverestControl,
    GenKwConfig,
    KnownResponseTypes,
    Observation,
    SurfaceConfig,
)
from ert.config import Field as FieldConfig
from ert.config._create_observation_dataframes import create_observation_dataframes
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_arg import create_run_arguments
from ert.run_models.run_model import RunModel, RunModelConfig
from ert.sample_prior import sample_prior
from ert.storage import DictEncodedDataFrame
from ert.storage.local_ensemble import LocalEnsemble


class InitialEnsembleRunModelConfig(RunModelConfig):
    experiment_name: str
    design_matrix: DictEncodedDataFrame | None
    parameter_configuration: list[
        Annotated[
            (GenKwConfig | SurfaceConfig | FieldConfig | EverestControl),
            Field(discriminator="type"),
        ]
    ]
    response_configuration: list[
        Annotated[
            (KnownResponseTypes),
            Field(discriminator="type"),
        ]
    ]
    ert_templates: list[tuple[str, str]]
    observations: list[Observation] | None = None


class InitialEnsembleRunModel(RunModel, InitialEnsembleRunModelConfig):
    def _sample_and_evaluate_ensemble(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        ensemble_storage: LocalEnsemble,
    ) -> LocalEnsemble:
        sample_prior(
            ensemble_storage,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in self.parameter_configuration],
            random_seed=self.random_seed,
            num_realizations=self.runpath_config.num_realizations,
            design_matrix_df=self.design_matrix.to_polars()
            if self.design_matrix is not None
            else None,
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

    def observation_dataframes(self) -> dict[str, pl.DataFrame]:
        if self.observations is None:
            return {}

        rft_config = next(
            (r for r in self.response_configuration if r.type == "rft"),
            None,
        )

        return create_observation_dataframes(
            observations=self.observations,
            rft_config=rft_config,
        )
