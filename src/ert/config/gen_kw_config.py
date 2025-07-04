from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast, overload

import networkx as nx
import numpy as np
import pandas as pd
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

    def sample_or_load(self, real_nr: int, random_seed: int) -> pl.DataFrame:
        keys = [e.name for e in self.transform_functions]
        parameter_value = self._sample_value(
            self.name,
            keys,
            str(random_seed),
            real_nr,
        )

        parameter_dict = {
            parameter.name: parameter_value[idx]
            for idx, parameter in enumerate(self.transform_functions)
        }
        parameter_dict["realization"] = real_nr
        return pl.DataFrame(
            parameter_dict,
            schema={tf.name: pl.Float64 for tf in self.transform_functions}
            | {"realization": pl.Int64},
        )

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

        log10_data: dict[str, float | str] = {
            tf.name: math.log10(data[tf.name])
            for tf in self.transform_functions
            if isinstance(tf.distribution, LogNormalSettings | LogUnifSettings)
            and isinstance(data[tf.name], (int, float))
        }

        if log10_data:
            return {self.name: data, f"LOG10_{self.name}": log10_data}
        else:
            return {self.name: data}

    def save_parameters(
        self,
        ensemble: Ensemble,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        parameter_dict = {
            parameter.name: data[idx]
            for idx, parameter in enumerate(self.transform_functions)
        }
        parameter_dict["realization"] = realization
        ensemble.save_parameters(
            self.name,
            realization=None,
            dataset=pl.DataFrame(
                parameter_dict,
                schema={tf.name: pl.Float64 for tf in self.transform_functions}
                | {"realization": pl.Int64},
            ),
        )

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        return (
            ensemble.load_parameters(self.name, realizations)
            .drop("realization")
            .to_numpy()
            .T.copy()
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

    def transform(self, array: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Transform the input array in accordance with priors

        Parameters:
            array: An array of standard normal values

        Returns: Transformed array, where each element has been transformed from
            a standard normal distribution to the distribution set by the user
        """
        array = np.array(array)
        for index, tf in enumerate(self.transform_functions):
            array[index] = tf.distribution.transform(array[index])
        return array

    def transform_col(self, param_name: str) -> Callable[[float], float]:
        tf: TransformFunction | None = None
        for tf in self.transform_functions:
            if tf.name == param_name:
                break
        assert tf is not None, f"Transform function {param_name} not found"
        return tf.distribution.transform

    @staticmethod
    def _values_from_file(file_name: str, keys: list[str]) -> npt.NDArray[np.double]:
        df = pd.read_csv(file_name, sep=r"\s+", header=None)
        # This means we have a key: value mapping in the
        # file otherwise it is just a list of values
        if df.shape[1] == 2:
            # We need to sort the user input keys by the
            # internal order of sub-parameters:
            df = df.set_index(df.columns[0])
            return df.reindex(keys).values.flatten()
        if not np.issubdtype(df.values.dtype, np.number):
            raise ValueError(
                f"The file {file_name} did not contain numbers, got {df.values.dtype}"
            )
        return df.values.flatten()

    @staticmethod
    def _sample_value(
        parameter_group_name: str,
        keys: list[str],
        global_seed: str,
        realization: int,
    ) -> npt.NDArray[np.double]:
        """
        Generate a sample value for each key in a parameter group.

        The sampling is reproducible and dependent on a global seed combined
        with the parameter group name and individual key names. The 'realization'
        parameter determines the specific sample point from the distribution for each
        parameter.

        Parameters:
        - parameter_group_name (str): The name of the parameter group, used to ensure
        unique RNG seeds for different groups.
        - keys (list[str]): A list of parameter keys for which the sample values are
        generated.
        - global_seed (str): A global seed string used for RNG seed generation to ensure
        reproducibility across runs.
        - realization (int): An integer used to advance the RNG to a specific point in
        its sequence, effectively selecting the 'realization'-th sample from the
        distribution.

        Returns:
        - npt.NDArray[np.double]: An array of sample values, one for each key in the
        provided list.

        Note:
        The method uses SHA-256 for hash generation and numpy's default random number
        generator for sampling. The RNG state is advanced to the 'realization' point
        before generating a single sample, enhancing efficiency by avoiding the
        generation of large, unused sample sets.
        """
        parameter_values = []
        for key in keys:
            key_hash = sha256(
                global_seed.encode("utf-8") + f"{parameter_group_name}:{key}".encode()
            )
            seed = np.frombuffer(key_hash.digest(), dtype="uint32")
            rng = np.random.default_rng(seed)

            # Advance the RNG state to the realization point
            rng.standard_normal(realization)

            # Generate a single sample
            value = rng.standard_normal(1)
            parameter_values.append(value[0])
        return np.array(parameter_values)

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
