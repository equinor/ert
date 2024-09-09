from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Union,
    no_type_check,
    overload,
)

from ert.field_utils import get_shape

from .field import Field
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig
from .parameter_config import ParameterConfig
from .parsing import ConfigDict, ConfigKeys, ConfigValidationError
from .refcase import Refcase
from .response_config import ResponseConfig
from .summary_config import SummaryConfig
from .surface_config import SurfaceConfig

_KNOWN_RESPONSE_TYPES = [SummaryConfig, GenDataConfig]

logger = logging.getLogger(__name__)


@overload
def _get_abs_path(file: None) -> None:
    pass


@overload
def _get_abs_path(file: str) -> str:
    pass


def _get_abs_path(file: Optional[str]) -> Optional[str]:
    if file is not None:
        file = os.path.realpath(file)
    return file


@dataclass
class EnsembleConfig:
    grid_file: Optional[str] = None
    response_configs: Dict[str, ResponseConfig] = field(default_factory=dict)
    parameter_configs: Dict[str, ParameterConfig] = field(default_factory=dict)
    refcase: Optional[Refcase] = None
    eclbase: Optional[str] = None

    def __post_init__(self) -> None:
        self._check_for_duplicate_names(
            [p.name for p in self.parameter_configs.values()],
            [key for config in self.response_configs.values() for key in config.keys],
        )

        self.grid_file = _get_abs_path(self.grid_file)

    @staticmethod
    def _check_for_duplicate_names(
        parameter_list: List[str], gen_data_list: List[str]
    ) -> None:
        names_counter = Counter(g for g in parameter_list + gen_data_list)
        duplicate_names = [n for n, c in names_counter.items() if c > 1]
        if duplicate_names:
            raise ConfigValidationError.with_context(
                "GEN_KW and GEN_DATA contained"
                f" duplicate name{'s' if len(duplicate_names) > 1 else ''}:"
                f" {','.join(duplicate_names)}",
                duplicate_names[0],
            )

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> EnsembleConfig:
        grid_file_path = config_dict.get(ConfigKeys.GRID)
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])
        dims = None
        if grid_file_path is not None:
            try:
                dims = get_shape(grid_file_path)
            except Exception as err:
                raise ConfigValidationError.with_context(
                    f"Could not read grid file {grid_file_path}: {err}",
                    grid_file_path,
                ) from err

        def make_field(field_list: List[str]) -> Field:
            if grid_file_path is None:
                raise ConfigValidationError.with_context(
                    "In order to use the FIELD keyword, a GRID must be supplied.",
                    field_list,
                )
            if dims is None:
                raise ConfigValidationError.with_context(
                    f"Grid file {grid_file_path} did not contain dimensions",
                    grid_file_path,
                )
            return Field.from_config_list(grid_file_path, dims, field_list)

        parameter_configs = (
            [GenKwConfig.from_config_list(g) for g in gen_kw_list]
            + [SurfaceConfig.from_config_list(s) for s in surface_list]
            + [make_field(f) for f in field_list]
        )

        response_configs: List[ResponseConfig] = []

        for config_cls in _KNOWN_RESPONSE_TYPES:
            instance = config_cls.from_config_dict(config_dict)

            if instance is not None and instance.keys:
                response_configs.append(instance)

        refcase = Refcase.from_config_dict(config_dict)
        eclbase = config_dict.get("ECLBASE")
        if eclbase is not None:
            eclbase = eclbase.replace("%d", "<IENS>")

        return cls(
            grid_file=grid_file_path,
            response_configs={response.name: response for response in response_configs},
            parameter_configs={
                parameter.name: parameter for parameter in parameter_configs
            },
            eclbase=eclbase,
            refcase=refcase,
        )

    def __getitem__(self, key: str) -> Union[ParameterConfig, ResponseConfig]:
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

    def addNode(self, config_node: Union[ParameterConfig, ResponseConfig]) -> None:
        assert config_node is not None
        if config_node.name in self:
            raise ConfigValidationError(
                f"Config node with key {config_node.name!r} already present in ensemble config"
            )

        if isinstance(config_node, ParameterConfig):
            logger.info(
                f"Adding {type(config_node).__name__} config (of size {len(config_node)}) to parameter_configs"
            )
            self.parameter_configs[config_node.name] = config_node
        else:
            self.response_configs[config_node.name] = config_node

    def get_keylist_gen_kw(self) -> List[str]:
        return [
            val.name
            for val in self.parameter_configuration
            if isinstance(val, GenKwConfig)
        ]

    @property
    def parameters(self) -> List[str]:
        return list(self.parameter_configs)

    @property
    def responses(self) -> List[str]:
        return list(self.response_configs)

    @property
    def keys(self) -> List[str]:
        return self.parameters + self.responses

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    @property
    def parameter_configuration(self) -> List[ParameterConfig]:
        return list(self.parameter_configs.values())

    @property
    def response_configuration(self) -> List[ResponseConfig]:
        return list(self.response_configs.values())
