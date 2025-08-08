from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Self, no_type_check, overload

from pydantic import BaseModel, Field, model_validator

from ert.field_utils import get_shape

from .everest_constraints_config import EverestConstraintsConfig
from .everest_objective_config import EverestObjectivesConfig
from .ext_param_config import ExtParamConfig
from .field import Field as FieldConfig
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig
from .parameter_config import ParameterConfig
from .parsing import ConfigDict, ConfigKeys, ConfigValidationError
from .refcase import Refcase
from .response_config import ResponseConfig
from .summary_config import SummaryConfig
from .surface_config import SurfaceConfig

_KNOWN_RESPONSE_TYPES = [
    SummaryConfig,
    GenDataConfig,
    EverestConstraintsConfig,
    EverestObjectivesConfig,
]

logger = logging.getLogger(__name__)


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


class EnsembleConfig(BaseModel):
    response_configs: dict[
        str,
        SummaryConfig
        | GenDataConfig
        | EverestConstraintsConfig
        | EverestObjectivesConfig,
    ] = Field(default_factory=dict)
    parameter_configs: dict[
        str, GenKwConfig | FieldConfig | SurfaceConfig | ExtParamConfig
    ] = Field(default_factory=dict)
    refcase: Refcase | None = None

    @model_validator(mode="after")
    def set_derived_fields(self) -> Self:
        self._check_for_duplicate_names(
            [p.name for p in self.parameter_configs.values()],
            [key for config in self.response_configs.values() for key in config.keys],
        )
        self._check_for_duplicate_gen_kw_param_names(
            [p for p in self.parameter_configs.values() if isinstance(p, GenKwConfig)]
        )

        return self

    @staticmethod
    def _check_for_duplicate_names(
        parameter_list: list[str], gen_data_list: list[str]
    ) -> None:
        names_counter = Counter(g for g in parameter_list + gen_data_list)
        duplicate_names = [n for n, c in names_counter.items() if c > 1]
        if duplicate_names:
            raise ConfigValidationError(
                "GEN_KW and GEN_DATA contained"
                f" duplicate name{'s' if len(duplicate_names) > 1 else ''}:"
                f" {','.join(duplicate_names)}",
            )

    @staticmethod
    def _check_for_duplicate_gen_kw_param_names(gen_kw_list: list[GenKwConfig]) -> None:
        gen_kw_param_count = Counter(
            keyword.name for p in gen_kw_list for keyword in p.transform_functions
        )
        duplicate_gen_kw_names = [
            (n, c) for n, c in gen_kw_param_count.items() if c > 1
        ]

        if duplicate_gen_kw_names:
            duplicates_formatted = ", ".join(
                f"{name}({count})" for name, count in duplicate_gen_kw_names
            )
            raise ConfigValidationError(
                "GEN_KW parameter names must be unique, found duplicates:"
                f" {duplicates_formatted}",
            )

    @no_type_check
    @staticmethod
    def get_gen_kw_templates(config_dict: ConfigDict) -> list[tuple[str, str]]:
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        return [
            template
            for g in gen_kw_list
            if (template := GenKwConfig.templates_from_config(g)) is not None
        ]

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> EnsembleConfig:
        # Grid file handling:
        # Each field can specify its own grid file, or fall back to a global grid.
        # If neither is provided, validation will fail when processing fields.
        global_grid_file_path = config_dict.get(ConfigKeys.GRID)
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])
        global_dims = None

        # When users specify GRID as a separate line in the config,
        # and not as an option to the FIELD keyword.
        if global_grid_file_path is not None:
            try:
                global_dims = get_shape(global_grid_file_path)
            except Exception as err:
                raise ConfigValidationError.with_context(
                    f"Could not read grid file {global_grid_file_path}: {err}",
                    global_grid_file_path,
                ) from err

        def make_field(field_list: list[str]) -> FieldConfig:
            # An example of `field_list` when the keyword `GRID` is set:
            # ['COND', 'PARAMETER', 'cond.bgrdecl',
            #  {'INIT_FILES': 'cond_%d.bgrdecl', 'FORWARD_INIT': 'False',
            #   'GRID': 'CASE.EGRID'}]
            # The fourth element (index 3) of this list is a dictionary of
            # optional keywords, one of which is `GRID`.
            field_settings = field_list[3]
            grid_file_path = field_settings.get(ConfigKeys.GRID)

            if grid_file_path is None and global_grid_file_path is None:
                raise ConfigValidationError.with_context(
                    "In order to use the FIELD keyword, a GRID must be supplied.",
                    field_list,
                )

            # Use field-specific grid if provided,
            # otherwise fall back to global grid.
            if grid_file_path is not None:
                try:
                    dims = get_shape(grid_file_path)
                except Exception as err:
                    raise ConfigValidationError.with_context(
                        f"Could not read grid file {grid_file_path}: {err}",
                        grid_file_path,
                    ) from err
            else:
                grid_file_path = global_grid_file_path
                dims = global_dims

            if dims is None:
                raise ConfigValidationError.with_context(
                    f"Grid file {grid_file_path} did not contain dimensions",
                    grid_file_path,
                )

            return FieldConfig.from_config_list(grid_file_path, dims, field_list)

        parameter_configs = (
            [GenKwConfig.from_config_list(g) for g in gen_kw_list]
            + [SurfaceConfig.from_config_list(s) for s in surface_list]
            + [make_field(f) for f in field_list]
        )

        response_configs: list[ResponseConfig] = []

        for config_cls in _KNOWN_RESPONSE_TYPES:
            instance = config_cls.from_config_dict(config_dict)

            if instance is not None and instance.keys:
                response_configs.append(instance)

        refcase = Refcase.from_config_dict(config_dict)

        return cls(
            response_configs={response.name: response for response in response_configs},
            parameter_configs={
                parameter.name: parameter for parameter in parameter_configs
            },
            refcase=refcase,
        )

    def __getitem__(self, key: str) -> ParameterConfig | ResponseConfig:
        if key in self.parameter_configs:
            return self.parameter_configs[key]
        elif key in self.response_configs:
            return self.response_configs[key]
        elif _config := next(
            (c for c in self.response_configs.values() if key in c.keys), None
        ):
            # Only hit by blockfs migration
            # returns the same config for one call per
            # response type. Is later deduped before saving to json
            return _config
        else:
            raise KeyError(f"The key:{key} is not in the ensemble configuration")

    def hasNodeGenData(self, key: str) -> bool:
        if "gen_data" not in self.response_configs:
            return False

        config = self.response_configs["gen_data"]
        return key in config.keys

    def get_keylist_gen_kw(self) -> list[str]:
        return [
            val.name
            for val in self.parameter_configuration
            if isinstance(val, GenKwConfig)
        ]

    def get_all_gen_kw_parameter_names(self) -> list[str]:
        return [
            parameter_key
            for p in self.parameter_configs.values()
            if isinstance(p, GenKwConfig)
            for parameter_key in p.parameter_keys
        ]

    @property
    def parameters(self) -> list[str]:
        return list(self.parameter_configs)

    @property
    def responses(self) -> list[str]:
        return list(self.response_configs)

    @property
    def keys(self) -> list[str]:
        return self.parameters + self.responses

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    @property
    def parameter_configuration(self) -> list[ParameterConfig]:
        return list(self.parameter_configs.values())

    @property
    def response_configuration(self) -> list[ResponseConfig]:
        return list(self.response_configs.values())
