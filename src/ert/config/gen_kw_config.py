from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast, overload

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator
from typing_extensions import TypedDict

from ._str_to_bool import str_to_bool
from .distribution import (
    DISTRIBUTION_CLASSES,
    ConstSettings,
    DerrfSettings,
    DUnifSettings,
    ErrfSettings,
    LogNormalSettings,
    LogUnifSettings,
    NormalSettings,
    RawSettings,
    TriangularSettings,
    TruncNormalSettings,
    UnifSettings,
    get_distribution,
)
from .parameter_config import ParameterConfig, ParameterMetadata
from .parsing import ConfigValidationError, ConfigWarning, ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


class PriorDict(TypedDict):
    key: str
    function: str
    parameters: dict[str, float]


@overload
def _get_abs_path(file: None) -> None:
    pass


@overload
def _get_abs_path(file: str) -> str:
    pass


def _get_abs_path(file: str | None) -> str | None:
    if file is not None:
        file = os.path.realpath(file)
    return file


class TransformFunctionDefinition(BaseModel):
    name: str
    param_name: str
    values: list[Any]


@dataclass
class TransformFunction:
    name: str
    distribution: Annotated[
        UnifSettings
        | LogNormalSettings
        | LogUnifSettings
        | DUnifSettings
        | RawSettings
        | ConstSettings
        | NormalSettings
        | TruncNormalSettings
        | ErrfSettings
        | DerrfSettings
        | TriangularSettings,
        Field(discriminator="name"),
    ]

    @property
    def parameter_list(self) -> dict[str, float]:
        """Return the parameters of the distribution as a dictionary."""
        return self.distribution.model_dump(exclude={"name"})


class GenKwConfig(ParameterConfig):
    type: Literal["gen_kw"] = "gen_kw"
    transform_function_definitions: list[TransformFunctionDefinition]

    _transform_functions: list[TransformFunction] = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_setup_transform_functions(self) -> Self:
        transform_functions: list[TransformFunction] = []

        errors = []
        for e in self.transform_function_definitions:
            try:
                if isinstance(e, dict):
                    transform_functions.append(
                        self._parse_transform_function_definition(
                            TransformFunctionDefinition(**e)
                        )
                    )
                else:
                    transform_functions.append(
                        self._parse_transform_function_definition(e)
                    )
            except ConfigValidationError as e:
                errors.append(e)

        self._transform_functions = transform_functions

        try:
            self._validate()
        except ConfigValidationError as e:
            errors.append(e)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        return self

    def __contains__(self, item: str) -> bool:
        return item in [v.name for v in self.transform_function_definitions]

    def __len__(self) -> int:
        return len(self.transform_functions)

    @property
    def transform_functions(self) -> list[TransformFunction]:
        return self._transform_functions

    @property
    def parameter_keys(self) -> list[str]:
        keys = []
        for tf in self.transform_functions:
            keys.append(tf.name)

        return keys

    @property
    def metadata(self) -> list[ParameterMetadata]:
        return [
            ParameterMetadata(
                key=f"{self.name}:{tf.name}",
                transformation=tf.distribution.name.upper(),
                dimensionality=1,
                userdata={"data_origin": "GEN_KW"},
            )
            for tf in self.transform_functions
        ]

    @classmethod
    def templates_from_config(
        cls, gen_kw: list[str | dict[str, str]]
    ) -> tuple[str, str] | None:
        gen_kw_key = cast(str, gen_kw[0])
        positional_args = cast(list[str], gen_kw[:-1])

        if len(positional_args) == 4:
            output_file = positional_args[2]
            parameter_file_context = positional_args[3][0]
            template_file = _get_abs_path(positional_args[1][0])
            if not os.path.isfile(template_file):
                raise ConfigValidationError.with_context(
                    f"No such template file: {template_file}", positional_args[1]
                )
            elif Path(template_file).stat().st_size == 0:
                token = getattr(parameter_file_context, "token", parameter_file_context)
                ConfigWarning.deprecation_warn(
                    f"The template file for GEN_KW ({gen_kw_key}) is empty. "
                    "If templating is not needed, you "
                    "can use GEN_KW with just the distribution file "
                    f"instead: GEN_KW {gen_kw_key} {token}",
                    positional_args[1],
                )
            if output_file.startswith("/"):
                raise ConfigValidationError.with_context(
                    f"Output file cannot have an absolute path {output_file}",
                    positional_args[2],
                )
            return template_file, output_file
        return None

    @classmethod
    def from_config_list(cls, gen_kw: list[str | dict[str, str]]) -> Self:
        gen_kw_key = cast(str, gen_kw[0])

        options = cast(dict[str, str], gen_kw[-1])
        positional_args = cast(list[str], gen_kw[:-1])
        errors = []
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        if _get_abs_path(options.get("INIT_FILES")):
            raise ConfigValidationError.with_context(
                "INIT_FILES with GEN_KW has been removed. "
                f"Please remove INIT_FILES from the GEN_KW {gen_kw_key} config. "
                "Alternatively, use DESIGN_MATRIX to load parameters from files.",
                gen_kw,
            )

        if len(positional_args) == 2:
            parameter_file_contents = positional_args[1][1]
            parameter_file_context = positional_args[1][0]
        elif len(positional_args) == 4:
            parameter_file_contents = positional_args[3][1]
            parameter_file_context = positional_args[3][0]
        else:
            raise ConfigValidationError(
                f"Unexpected positional arguments: {positional_args}"
            )

        transform_function_definitions: list[TransformFunctionDefinition] = []
        for line_number, item in enumerate(parameter_file_contents.splitlines()):
            item = item.split("--")[0]  # remove comments
            if item.strip():  # only lines with content
                items = item.split()
                if len(items) < 2:
                    errors.append(
                        ConfigValidationError.with_context(
                            f"Too few values on line {line_number} in parameter "
                            f"file {parameter_file_context}",
                            gen_kw,
                        )
                    )
                else:
                    transform_function_definitions.append(
                        TransformFunctionDefinition(
                            name=items[0],
                            param_name=items[1],
                            values=items[2:],
                        )
                    )
        if not transform_function_definitions:
            errors.append(
                ConfigValidationError.with_context(
                    f"No parameters specified in {parameter_file_context}",
                    parameter_file_context,
                )
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        if gen_kw_key == "PRED" and update_parameter:
            ConfigWarning.warn(
                "GEN_KW PRED used to hold a special meaning and be "
                "excluded from being updated.\n If the intention was "
                "to exclude this from updates, set UPDATE:FALSE.\n",
                gen_kw_key,
            )
        try:
            return cls(
                name=gen_kw_key,
                forward_init=False,
                transform_function_definitions=transform_function_definitions,
                update=update_parameter,
            )
        except ValidationError as e:
            raise ConfigValidationError.from_pydantic(e, gen_kw) from e

    def _validate(self) -> None:
        errors = []
        unique_keys = set()
        for prior in self.get_priors():
            key = prior["key"]
            if key in unique_keys:
                errors.append(
                    ErrorInfo(
                        f"Duplicate GEN_KW keys {key!r} found, keys must be unique."
                    ).set_context(self.name)
                )
            unique_keys.add(key)

        if errors:
            raise ConfigValidationError.from_collected(errors)

    def load_parameter_graph(self) -> nx.Graph[int]:
        # Create a graph with no edges
        graph_independence: nx.Graph[int] = nx.Graph()
        graph_independence.add_nodes_from(range(len(self.transform_functions)))
        return graph_independence

    def read_from_runpath(
        self,
        run_path: Path,
        real_nr: int,
        iteration: int,
    ) -> xr.Dataset:
        raise NotImplementedError

    def write_to_runpath(
        self,
        run_path: Path,
        real_nr: int,
        ensemble: Ensemble,
    ) -> dict[str, dict[str, float | str]]:
        df = ensemble.load_parameters(self.name, real_nr, transformed=True).drop(
            "realization"
        )

        assert isinstance(df, pl.DataFrame)
        if not df.width == len(self.transform_functions):
            raise ValueError(
                f"The configuration of GEN_KW parameter {self.name}"
                f" has {len(self.transform_functions)} parameters, but ensemble dataset"
                f" for realization {real_nr} has {df.width} parameters."
            )

        data = df.to_dicts()[0]
        return {self.name: data}

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        return (
            ensemble.load_parameters(self.name, realizations)
            .drop("realization")
            .to_numpy()
            .T.copy()
        )

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[int | None, pl.DataFrame]]:
        yield (
            None,
            pl.DataFrame(
                {
                    "realization": iens_active_index,
                }
            ).with_columns(
                [
                    pl.Series(from_data[i, :]).alias(param_name.name)
                    for i, param_name in enumerate(self.transform_functions)
                ]
            ),
        )

    def copy_parameters(
        self,
        source_ensemble: Ensemble,
        target_ensemble: Ensemble,
        realizations: npt.NDArray[np.int_],
    ) -> None:
        df = source_ensemble.load_parameters(self.name, realizations)
        target_ensemble.save_parameters(self.name, realization=None, dataset=df)

    def shouldUseLogScale(self, keyword: str) -> bool:
        for tf in self.transform_functions:
            if tf.name == keyword:
                return isinstance(tf.distribution, LogNormalSettings | LogUnifSettings)
        return False

    def get_priors(self) -> list[PriorDict]:
        priors: list[PriorDict] = []
        for tf in self.transform_functions:
            priors.append(
                {
                    "key": tf.name,
                    "function": tf.distribution.name.upper(),
                    "parameters": {
                        k.upper(): v
                        for k, v in tf.parameter_list.items()
                        if k != "name"
                    },
                }
            )
        return priors

    def transform_col(self, param_name: str) -> Callable[[float], float]:
        tf: TransformFunction | None = None
        for tf in self.transform_functions:
            if tf.name == param_name:
                break
        assert tf is not None, f"Transform function {param_name} not found"
        return tf.distribution.transform

    def _parse_transform_function_definition(
        self,
        t: TransformFunctionDefinition,
    ) -> TransformFunction:
        if t.param_name not in DISTRIBUTION_CLASSES:
            raise ConfigValidationError(
                f"Unknown distribution provided: {t.param_name}, for variable {t.name}",
                self.name,
            )

        cls = DISTRIBUTION_CLASSES[t.param_name]

        if len(t.values) != len(cls.get_param_names()):
            raise ConfigValidationError.with_context(
                f"Incorrect number of values: {t.values}, provided for variable "
                f"{t.name} with distribution {t.param_name}.",
                self.name,
            )
        param_floats = []
        for p in t.values:
            try:
                param_floats.append(float(p))
            except ValueError as e:
                raise ConfigValidationError.with_context(
                    f"Unable to convert '{p}' to float number for variable "
                    f"{t.name} with distribution {t.param_name}.",
                    self.name,
                ) from e
        try:
            dist = get_distribution(t.param_name, param_floats)
        except ValidationError as e:
            error_to_raise = ConfigValidationError.from_pydantic(
                error=e, context=self.name
            )
            for error_info in error_to_raise.errors:
                error_info.message += f" parameter {t.name}"

            raise error_to_raise from e

        return TransformFunction(name=t.name, distribution=dist)
