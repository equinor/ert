from __future__ import annotations

import os
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import ValidationError
from typing_extensions import TypedDict

from ._str_to_bool import str_to_bool
from .distribution import DISTRIBUTION_CLASSES, DistributionSettings, get_distribution
from .parameter_config import ParameterCardinality, ParameterConfig
from .parsing import ConfigValidationError, ConfigWarning

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


class DataSource(StrEnum):
    DESIGN_MATRIX = "design_matrix"
    SAMPLED = "sampled"


class GenKwConfig(ParameterConfig):
    type: Literal["gen_kw"] = "gen_kw"
    dimensionality: Literal[1] = 1
    distribution: DistributionSettings
    forward_init: bool = False
    update: bool = True
    group: str | None = None
    input_source: DataSource = DataSource.SAMPLED

    def __contains__(self, item: str) -> bool:
        return item == self.name

    def __len__(self) -> int:
        return 1

    @property
    def parameter_keys(self) -> list[str]:
        return [self.name]

    @property
    def cardinality(self) -> ParameterCardinality:
        return ParameterCardinality.multiple_configs_per_ensemble_dataset

    @property
    def group_name(self) -> str | None:
        return self.group

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
    def from_config_list(cls, config_list: list[str | dict[str, str]]) -> list[Self]:
        gen_kw_key = cast(str, config_list[0])

        options = cast(dict[str, str], config_list[-1])
        positional_args = cast(list[str | list[str]], config_list[:-1])
        errors = []
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        if _get_abs_path(options.get("INIT_FILES")):
            raise ConfigValidationError.with_context(
                "INIT_FILES with GEN_KW has been removed. "
                f"Please remove INIT_FILES from the GEN_KW {gen_kw_key} config. "
                "Alternatively, use DESIGN_MATRIX to load parameters from files.",
                config_list,
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

        distributions_spec: list[list[str]] = []
        for line_number, item in enumerate(parameter_file_contents.splitlines()):
            item = item.split("--")[0]  # remove comments
            if item.strip():  # only lines with content
                items = item.split()
                if len(items) < 2:
                    errors.append(
                        ConfigValidationError.with_context(
                            f"Too few values on line {line_number} in parameter "
                            f"file {parameter_file_context}",
                            config_list,
                        )
                    )
                else:
                    distributions_spec.append(items)

        if not distributions_spec:
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
            return [
                cls(
                    name=params[0],
                    group=gen_kw_key,
                    distribution=GenKwConfig._parse_distribution(
                        params[0], params[1], params[2:]
                    ),
                    forward_init=False,
                    update="CONST" not in params and update_parameter,
                )
                for params in distributions_spec
            ]
        except ConfigValidationError as e:
            raise ConfigValidationError.from_collected(
                [err.set_context(gen_kw_key) for err in e.errors]
            ) from e
        except ValidationError as e:
            raise ConfigValidationError.from_pydantic(e, config_list) from e

    def load_parameter_graph(self) -> nx.Graph[int]:
        # Create a graph with no edges
        graph_independence: nx.Graph[int] = nx.Graph()
        graph_independence.add_nodes_from([0])
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
        raise NotImplementedError

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
                    self.name: pl.Series(from_data.flatten()),
                }
            ),
        )

    def get_priors(self) -> list[PriorDict]:
        dist_json = self.distribution.model_dump(exclude={"name"})
        return [
            {
                "key": self.name,
                "function": self.distribution.name.upper(),
                "parameters": {k.upper(): v for k, v in dist_json.items()},
            }
        ]

    def transform_numpy(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.distribution.transform_numpy(x)

    @classmethod
    def _parse_distribution(
        cls, param_name: str, dist_name: str, values: list[str]
    ) -> DistributionSettings:
        if dist_name not in DISTRIBUTION_CLASSES:
            raise ConfigValidationError(
                f"Unknown distribution provided: {dist_name}"
                f", for variable {param_name}",
                param_name,
            )
        dist_cls = DISTRIBUTION_CLASSES[dist_name]

        if len(values) != len(dist_cls.get_param_names()):
            raise ConfigValidationError.with_context(
                f"Incorrect number of values: {values}, provided for variable "
                f"{param_name} with distribution {dist_name}.",
                param_name,
            )
        param_floats = []
        for p in values:
            try:
                param_floats.append(float(p))
            except ValueError as e:
                raise ConfigValidationError.with_context(
                    f"Unable to convert '{p}' to float number for variable "
                    f"{param_name} with distribution {dist_name}.",
                    param_name,
                ) from e
        try:
            dist = get_distribution(dist_name, param_floats)
        except ValidationError as e:
            error_to_raise = ConfigValidationError.from_pydantic(
                error=e, context=param_name
            )
            for error_info in error_to_raise.errors:
                error_info.message += f" parameter {param_name}"

            raise error_to_raise from e
        return dist
