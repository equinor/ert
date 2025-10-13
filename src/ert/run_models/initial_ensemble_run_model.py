from abc import ABC
from typing import Annotated, Any, Literal, Self

import numpy as np
import polars as pl
from polars.datatypes import DataTypeClass
from pydantic import BaseModel, Field, field_validator

from ert.config import (
    EverestConstraintsConfig,
    EverestObjectivesConfig,
    ExtParamConfig,
    GenDataConfig,
    GenKwConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config import Field as FieldConfig
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_arg import create_run_arguments
from ert.run_models.run_model import RunModel
from ert.sample_prior import sample_prior
from ert.storage.local_ensemble import LocalEnsemble


# https://github.com/pola-rs/polars/issues/13152#issuecomment-1864600078
# PS: Serializing/deserializing schema is scheduled to be added to polars core,
# ref https://github.com/pola-rs/polars/issues/20426
# then this workaround can be omitted.
def str_to_dtype(dtype_str: str) -> pl.DataType:
    dtype = eval(f"pl.{dtype_str}")
    if isinstance(dtype, DataTypeClass):
        dtype = dtype()
    return dtype


class DictEncodedDataFrame(BaseModel):
    type: Literal["dicts"]
    data: list[dict[str, Any]]
    datatypes: dict[str, str]

    @classmethod
    def from_polars(cls, data: pl.DataFrame) -> Self:
        str_schema = {k: str(dtype) for k, dtype in data.schema.items()}
        return cls(type="dicts", data=data.to_dicts(), datatypes=str_schema)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dicts(
            self.data,
            schema={
                col: str_to_dtype(dtype_str)
                for col, dtype_str in self.datatypes.items()
            },
        )


class InitialEnsembleRunModel(RunModel, ABC):
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
    observations: dict[str, DictEncodedDataFrame] | None

    @field_validator("observations", mode="before")
    def make_dict_encoded_observations(
        cls, v: dict[str, pl.DataFrame | DictEncodedDataFrame] | None
    ) -> dict[str, DictEncodedDataFrame] | None:
        if v is None:
            return None
        return {
            k: df
            if isinstance(df, DictEncodedDataFrame)
            else DictEncodedDataFrame.from_polars(df)
            for k, df in v.items()
        }

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
