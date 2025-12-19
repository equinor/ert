from typing import Annotated, Any

import numpy as np
import polars as pl
from pydantic import Field, field_validator

from ert.config import (
    ExtParamConfig,
    GenKwConfig,
    KnownResponseTypes,
    SurfaceConfig,
)
from ert.config import Field as FieldConfig
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
            (GenKwConfig | SurfaceConfig | FieldConfig | ExtParamConfig),
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
    observations: dict[str, DictEncodedDataFrame] | None = None

    @field_validator("observations", mode="before")
    @classmethod
    def make_dict_encoded_observations(
        cls, v: dict[str, pl.DataFrame | DictEncodedDataFrame | dict[str, Any]] | None
    ) -> dict[str, DictEncodedDataFrame] | None:
        if v is None:
            return None

        encoded = {}
        for k, df in v.items():
            match df:
                case DictEncodedDataFrame():
                    encoded[k] = df
                case pl.DataFrame():
                    encoded[k] = DictEncodedDataFrame.from_polars(df)
                case dict():
                    encoded[k] = DictEncodedDataFrame.model_validate(df)

        return encoded


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
