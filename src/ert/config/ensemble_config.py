from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, no_type_check, overload

from ecl.summary import EclSum

from ert.field_utils import get_shape

from .field import Field
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig
from .parameter_config import ParameterConfig
from .parsing import ConfigDict, ConfigKeys, ConfigValidationError
from .response_config import ResponseConfig
from .summary_config import SummaryConfig
from .surface_config import SurfaceConfig

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


class EnsembleConfig:
    @staticmethod
    def _load_refcase(refcase_file: Optional[str]) -> Optional[EclSum]:
        if refcase_file is None:
            return None

        refcase_filepath = Path(refcase_file)
        refcase_file = str(refcase_filepath.parent / refcase_filepath.stem)

        if not os.path.exists(refcase_file + ".UNSMRY"):
            raise ConfigValidationError(
                f"Cannot find UNSMRY file for refcase provided! {refcase_file}.UNSMRY"
            )

        if not os.path.exists(refcase_file + ".SMSPEC"):
            raise ConfigValidationError(
                f"Cannot find SMSPEC file for refcase provided! {refcase_file}.SMSPEC"
            )

        # defaults for loading refcase - necessary for using the function
        # exposed in python part of ecl
        refcase_load_args = {
            "load_case": refcase_file,
            "join_string": ":",
            "include_restart": True,
            "lazy_load": False,
            "file_options": 0,
        }
        return EclSum(**refcase_load_args)

    def __init__(  # noqa: 501 pylint: disable=too-many-arguments
        self,
        grid_file: Optional[str] = None,
        ref_case_file: Optional[str] = None,
        gen_data_list: Optional[List[GenDataConfig]] = None,
        gen_kw_list: Optional[List[GenKwConfig]] = None,
        surface_list: Optional[List[SurfaceConfig]] = None,
        summary_list: Optional[List[List[str]]] = None,
        field_list: Optional[List[Field]] = None,
        ecl_base: Optional[str] = None,
    ) -> None:
        _gen_kw_list = [] if gen_kw_list is None else gen_kw_list
        _gen_data_list = [] if gen_data_list is None else gen_data_list
        _surface_list = [] if surface_list is None else surface_list
        _field_list = [] if field_list is None else field_list
        _summary_list = [] if summary_list is None else summary_list

        self._check_for_duplicate_names(_gen_kw_list, _gen_data_list)

        self._grid_file = _get_abs_path(grid_file)
        self._refcase_file = _get_abs_path(ref_case_file)
        self.refcase: Optional[EclSum] = self._load_refcase(self._refcase_file)
        self.parameter_configs: Dict[str, ParameterConfig] = {}
        self.response_configs: Dict[str, ResponseConfig] = {}

        for gen_data in _gen_data_list:
            self.addNode(gen_data)

        for gen_kw in _gen_kw_list:
            self.addNode(gen_kw)

        for surface in _surface_list:
            self.addNode(surface)

        if ecl_base:
            ecl_base = ecl_base.replace("%d", "<IENS>")
            summary_keys = [item for sublist in _summary_list for item in sublist]
            self.add_summary_full(ecl_base, summary_keys, self.refcase)

        for field in _field_list:
            self.addNode(field)

    @staticmethod
    def _check_for_duplicate_names(
        gen_kw_list: List[GenKwConfig], gen_data_list: List[GenDataConfig]
    ) -> None:
        names_counter = Counter(g.name for g in gen_kw_list + gen_data_list)
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
        refcase_file_path = config_dict.get(ConfigKeys.REFCASE)
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        summary_list = config_dict.get(ConfigKeys.SUMMARY, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])
        dims = None
        if grid_file_path is not None:
            dims = get_shape(grid_file_path)

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

        return cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            gen_data_list=[GenDataConfig.from_config_list(g) for g in gen_data_list],
            gen_kw_list=[GenKwConfig.from_config_list(g) for g in gen_kw_list],
            surface_list=[SurfaceConfig.from_config_list(s) for s in surface_list],
            summary_list=summary_list,
            field_list=[make_field(f) for f in field_list],
            ecl_base=config_dict.get("ECLBASE"),
        )

    def _node_info(self, object_type: Type[Any]) -> str:
        key_list = self.getKeylistFromImplType(object_type)
        return f"{object_type}: " f"{[self[key] for key in key_list]}, "

    def __repr__(self) -> str:
        return (
            "EnsembleConfig(config_dict={"
            + self._node_info(GenDataConfig)
            + self._node_info(GenKwConfig)
            + self._node_info(SurfaceConfig)
            + self._node_info(SummaryConfig)
            + self._node_info(Field)
            + f"{ConfigKeys.GRID}: {self._grid_file},"
            + f"{ConfigKeys.REFCASE}: {self._refcase_file}"
            + "}"
        )

    def __getitem__(self, key: str) -> Union[ParameterConfig, ResponseConfig]:
        if key in self.parameter_configs:
            return self.parameter_configs[key]
        elif key in self.response_configs:
            return self.response_configs[key]
        else:
            raise KeyError(f"The key:{key} is not in the ensemble configuration")

    def getNodeGenData(self, key: str) -> GenDataConfig:
        gen_node = self.response_configs[key]
        assert isinstance(gen_node, GenDataConfig)
        return gen_node

    def hasNodeGenData(self, key: str) -> bool:
        return key in self.response_configs and isinstance(
            self.response_configs[key], GenDataConfig
        )

    def getNode(self, key: str) -> Union[ParameterConfig, ResponseConfig]:
        return self[key]

    def add_summary_full(
        self, ecl_base: str, key_list: List[str], refcase: Optional[EclSum]
    ) -> None:
        optional_keys = []
        for key in key_list:
            if "*" in key and refcase:
                optional_keys.extend(list(refcase.keys(pattern=key)))
            else:
                optional_keys.append(key)
        self.addNode(
            SummaryConfig(
                name="summary",
                input_file=ecl_base,
                keys=optional_keys,
                refcase=refcase,
            )
        )

    def check_unique_node(self, key: str) -> None:
        if key in self:
            raise ConfigValidationError(
                f"Config node with key {key!r} already present in ensemble config"
            )

    def addNode(self, config_node: Union[ParameterConfig, ResponseConfig]) -> None:
        assert config_node is not None
        self.check_unique_node(config_node.name)
        if isinstance(config_node, ParameterConfig):
            self.parameter_configs[config_node.name] = config_node
        else:
            self.response_configs[config_node.name] = config_node

    def getKeylistFromImplType(self, node_type: Type[Any]) -> List[str]:
        mylist = []

        for key in self.keys:
            if isinstance(self[key], node_type):
                mylist.append(key)

        return mylist

    def get_keylist_gen_kw(self) -> List[str]:
        return self.getKeylistFromImplType(GenKwConfig)

    def get_keylist_gen_data(self) -> List[str]:
        return self.getKeylistFromImplType(GenDataConfig)

    @property
    def grid_file(self) -> Optional[str]:
        return self._grid_file

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnsembleConfig):
            return False

        if self.keys != other.keys:
            return False

        for par in self.keys:
            if par in self and par in other:
                if self[par] != other[par]:
                    return False
            else:
                return False

        if (
            self._grid_file != other._grid_file
            or self._refcase_file != other._refcase_file
        ):
            return False

        return True

    def get_summary_keys(self) -> List[str]:
        if "summary" in self:
            summary = self["summary"]
            if isinstance(summary, SummaryConfig):
                return sorted(set(summary.keys))
        return []

    @property
    def parameter_configuration(self) -> List[ParameterConfig]:
        return list(self.parameter_configs.values())
