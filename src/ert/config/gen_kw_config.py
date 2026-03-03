from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, cast, overload

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


# Represents one parsed GEN_KW config line as produced by the lark parser.
# Each element is either a plain string, an EXISTING_PATH_INLINE tuple
# (resolved_path, file_contents), or the trailing options dict.
GenKwConfigList: TypeAlias = list[str | tuple[str, str] | dict[str, str]]


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


@dataclass(frozen=True)
class GenKwOptions:
    """Typed representation of GEN_KW keyword options."""

    update: bool = True
    # Deprecated – only kept to produce a helpful migration error.
    init_files: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> GenKwOptions:
        return cls(
            update=str_to_bool(raw.get("UPDATE", "TRUE")),
            init_files=raw.get("INIT_FILES"),
        )


@dataclass(frozen=True)
class _ParsedGenKwConfig:
    """Structured representation of a parsed GEN_KW config line.

    The three ``*_context`` fields typed as ``Any``
    (``template_context``, ``output_context``, ``source_context``) are
    opaque parser tokens (``FileContextToken`` / ``ContextList``) whose
    sole purpose is to attach file/line information to error messages.
    They are typed as ``Any`` because the error-reporting API
    (``ErrorInfo.set_context``) already accepts ``Any``.
    """

    gen_kw_key: str
    options: GenKwOptions
    parameter_file_context: str
    parameter_file_contents: str
    template_file: str | None
    output_file: str | None
    # Opaque error-reporting tokens (FileContextToken / ContextList)
    template_context: Any = None
    output_context: Any = None
    source_context: Any = None


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
        cls, config_list: GenKwConfigList
    ) -> tuple[str, str] | None:
        return cls._templates_from_parsed(cls._parse_from_config_list(config_list))

    @classmethod
    def _templates_from_parsed(
        cls, parsed: _ParsedGenKwConfig
    ) -> tuple[str, str] | None:
        if parsed.template_file is None or parsed.output_file is None:
            return None
        cls._validate_template_and_output(
            gen_kw_key=parsed.gen_kw_key,
            template_file=parsed.template_file,
            output_file=parsed.output_file,
            parameter_file_context=parsed.parameter_file_context,
            template_context=parsed.template_context,
            output_context=parsed.output_context,
        )
        return parsed.template_file, parsed.output_file

    @classmethod
    def from_config_list(cls, config_list: GenKwConfigList) -> list[Self]:
        parsed = cls._parse_from_config_list(config_list)
        gen_kw_key = parsed.gen_kw_key
        errors = []
        if parsed.options.init_files:
            raise ConfigValidationError.with_context(
                "INIT_FILES with GEN_KW has been removed. "
                f"Please remove INIT_FILES from the GEN_KW {gen_kw_key} config. "
                "Alternatively, use DESIGN_MATRIX to load parameters from files.",
                parsed.source_context,
            )

        distributions_spec: list[list[str]] = []
        for line_number, item in enumerate(parsed.parameter_file_contents.splitlines()):
            item = item.split("--")[0]  # remove comments
            if item.strip():  # only lines with content
                items = item.split()
                if len(items) < 2:
                    errors.append(
                        ConfigValidationError.with_context(
                            f"Too few values on line {line_number} in parameter "
                            f"file {parsed.parameter_file_context}",
                            parsed.source_context,
                        )
                    )
                else:
                    distributions_spec.append(items)

        if not distributions_spec:
            errors.append(
                ConfigValidationError.with_context(
                    f"No parameters specified in {parsed.parameter_file_context}",
                    parsed.parameter_file_context,
                )
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        if gen_kw_key == "PRED" and parsed.options.update:
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
                    update="CONST" not in params and parsed.options.update,
                )
                for params in distributions_spec
            ]
        except ConfigValidationError as e:
            raise ConfigValidationError.from_collected(
                [err.set_context(gen_kw_key) for err in e.errors]
            ) from e
        except ValidationError as e:
            raise ConfigValidationError.from_pydantic(e, parsed.source_context) from e

    @classmethod
    def _parse_from_config_list(
        cls, config_list: GenKwConfigList
    ) -> _ParsedGenKwConfig:
        # config_list layout:
        #   2-arg: [KEY, (param_path, param_contents), options_dict]
        #   4-arg: [KEY, (template_path, _), output_file,
        #               (param_path, param_contents), options_dict]
        gen_kw_key = cast(str, config_list[0])
        options = GenKwOptions.from_raw(cast(dict[str, str], config_list[-1]))
        positional_args = config_list[:-1]

        if len(positional_args) == 2:
            param_file = cast(tuple[str, str], positional_args[1])
            parameter_file_context = param_file[0]
            parameter_file_contents = param_file[1]
            return _ParsedGenKwConfig(
                gen_kw_key=gen_kw_key,
                options=options,
                parameter_file_context=parameter_file_context,
                parameter_file_contents=parameter_file_contents,
                template_file=None,
                output_file=None,
                source_context=config_list,
            )

        if len(positional_args) == 4:
            template_tuple = cast(tuple[str, str], positional_args[1])
            template_file = _get_abs_path(template_tuple[0])
            assert template_file is not None
            output_file = cast(str, positional_args[2])
            param_file = cast(tuple[str, str], positional_args[3])
            parameter_file_context = param_file[0]
            parameter_file_contents = param_file[1]
            return _ParsedGenKwConfig(
                gen_kw_key=gen_kw_key,
                options=options,
                parameter_file_context=parameter_file_context,
                parameter_file_contents=parameter_file_contents,
                template_file=template_file,
                output_file=output_file,
                template_context=template_tuple[0],
                output_context=positional_args[2],
                source_context=config_list,
            )

        raise ConfigValidationError(
            f"Unexpected positional arguments: {positional_args}"
        )

    @classmethod
    def _validate_template_and_output(
        cls,
        gen_kw_key: str,
        template_file: str,
        output_file: str,
        parameter_file_context: str,
        template_context: str | tuple[str, str] | dict[str, str],
        output_context: str | tuple[str, str] | dict[str, str],
    ) -> None:
        if not os.path.isfile(template_file):
            raise ConfigValidationError.with_context(
                f"No such template file: {template_file}", template_context
            )
        if Path(template_file).stat().st_size == 0:
            token = getattr(parameter_file_context, "token", parameter_file_context)
            ConfigWarning.deprecation_warn(
                f"The template file for GEN_KW ({gen_kw_key}) is empty. "
                "If templating is not needed, you "
                "can use GEN_KW with just the distribution file "
                f"instead: GEN_KW {gen_kw_key} {token}",
                template_context,
            )
        if output_file.startswith("/"):
            raise ConfigValidationError.with_context(
                f"Output file cannot have an absolute path {output_file}",
                output_context,
            )

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
