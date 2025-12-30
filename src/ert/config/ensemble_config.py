from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from .everest_control import EverestControl
from .field import Field as FieldConfig
from .gen_kw_config import GenKwConfig
from .known_response_types import KNOWN_ERT_RESPONSE_TYPES, KnownErtResponseTypes
from .parameter_config import ParameterConfig
from .parsing import ConfigDict, ConfigKeys, ConfigValidationError
from .response_config import ResponseConfig
from .surface_config import SurfaceConfig

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    response_configs: dict[str, KnownErtResponseTypes] = Field(default_factory=dict)
    parameter_configs: dict[
        str, GenKwConfig | FieldConfig | SurfaceConfig | EverestControl
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_derived_fields(self) -> Self:
        self._check_for_duplicate_names(
            [p.name for p in self.parameter_configs.values()],
            [key for config in self.response_configs.values() for key in config.keys],
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
        gen_kw_param_count = Counter(p.name for p in gen_kw_list)
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

    @staticmethod
    def get_gen_kw_templates(config_dict: ConfigDict) -> list[tuple[str, str]]:
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        return [
            template
            for g in gen_kw_list
            if (template := GenKwConfig.templates_from_config(g)) is not None
        ]

    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> EnsembleConfig:
        # Grid file handling:
        # Each field can specify its own grid file, or fall back to a global grid.
        # If neither is provided, validation will fail when processing fields.
        global_grid_file_path = config_dict.get(ConfigKeys.GRID)
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])

        if global_grid_file_path is not None:
            global_grid_file_path = Path(global_grid_file_path)

            grid_extension = global_grid_file_path.suffix.lower()
            if grid_extension not in {".egrid", ".grid"}:
                raise ConfigValidationError("Only EGRID and GRID formats are supported")

        def make_field(field_list: list[str | dict[str, str]]) -> FieldConfig:
            # An example of `field_list` when the keyword `GRID` is set:
            # ['COND', 'PARAMETER', 'cond.bgrdecl',
            #  {'INIT_FILES': 'cond_%d.bgrdecl', 'FORWARD_INIT': 'False',
            #   'GRID': 'CASE.EGRID'}]
            # The fourth element (index 3) of this list is a dictionary of
            # optional keywords, one of which is `GRID`.
            field_settings = field_list[3]
            assert isinstance(field_settings, dict)
            grid_file_path = field_settings.get(ConfigKeys.GRID)

            if grid_file_path is None and global_grid_file_path is None:
                raise ConfigValidationError.with_context(
                    "In order to use the FIELD keyword, a GRID must be supplied.",
                    field_list,
                )

            # Use field-specific grid if provided,
            # otherwise fall back to global grid.
            if grid_file_path is None:
                grid_file_path = str(global_grid_file_path)

            return FieldConfig.from_config_list(grid_file_path, field_list)

        gen_kw_cfgs = [
            cfg for g in gen_kw_list for cfg in GenKwConfig.from_config_list(g)
        ]

        parameter_configs = (
            gen_kw_cfgs
            + [SurfaceConfig.from_config_list(s) for s in surface_list]
            + [make_field(f) for f in field_list]
        )
        EnsembleConfig._check_for_duplicate_gen_kw_param_names(gen_kw_cfgs)
        response_configs: list[KnownErtResponseTypes] = []

        for config_cls in KNOWN_ERT_RESPONSE_TYPES:
            instance = config_cls.from_config_dict(config_dict)

            if instance is not None and instance.keys:
                response_configs.append(instance)

        return cls(
            response_configs={response.name: response for response in response_configs},
            parameter_configs={
                parameter.name: parameter for parameter in parameter_configs
            },
        )

    def __getitem__(self, key: str) -> ParameterConfig | ResponseConfig:
        if key in self.parameter_configs:
            return self.parameter_configs[key]
        elif key in self.response_configs:
            return self.response_configs[key]
        elif config := next(
            (c for c in self.response_configs.values() if key in c.keys), None
        ):
            # Only hit by blockfs migration
            # returns the same config for one call per
            # response type. Is later deduped before saving to json
            return config
        else:
            raise KeyError(f"The key:{key} is not in the ensemble configuration")

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
