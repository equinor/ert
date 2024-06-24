from __future__ import annotations

import logging
import os
from collections import Counter
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    no_type_check,
    overload,
)

from ert.config.commons import Refcase
from ert.config.field import Field
from ert.config.gen_kw_config import GenKwConfig
from ert.config.parameter_config import ParameterConfig
from ert.config.parsing import ConfigDict, ConfigKeys, ConfigValidationError
from ert.config.responses._read_summary import read_summary
from ert.config.responses.gen_data_config import GenDataConfig
from ert.config.responses.response_config import (
    ObservationConfig,
    ResponseConfig,
    ResponseConfigWithLifecycleHooks,
)
from ert.config.responses.summary_config import SummaryConfig
from ert.config.surface_config import SurfaceConfig
from ert.field_utils import get_shape

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
    def __init__(
        self,
        grid_file: Optional[str] = None,
        response_configs: Dict[str, ResponseConfigWithLifecycleHooks] = None,
        observation_configs: Dict[str, ObservationConfig] = None,
        gendata_list: Optional[List[GenDataConfig]] = None,
        genkw_list: Optional[List[GenKwConfig]] = None,
        surface_list: Optional[List[SurfaceConfig]] = None,
        eclbase: Optional[str] = None,
        field_list: Optional[List[Field]] = None,
        refcase: Optional[Refcase] = None,
    ) -> None:
        _genkw_list = [] if genkw_list is None else genkw_list
        _gendata_list = [] if gendata_list is None else gendata_list
        _surface_list = [] if surface_list is None else surface_list
        _field_list = [] if field_list is None else field_list

        self._check_for_duplicate_names(_genkw_list, _gendata_list)

        self._grid_file = _get_abs_path(grid_file)
        self.parameter_configs: Dict[str, ParameterConfig] = {}
        self.response_configs: Dict[
            str, Union[ResponseConfigWithLifecycleHooks, ResponseConfig]
        ] = response_configs  # TODO use only ported when porting is done
        self.observation_configs: Dict[str, ObservationConfig] = observation_configs
        self.refcase = refcase
        self.eclbase = eclbase

        for gen_data in _gendata_list:
            self.addNode(gen_data)

        for gen_kw in _genkw_list:
            self.addNode(gen_kw)

        for surface in _surface_list:
            self.addNode(surface)

        for field in _field_list:
            self.addNode(field)

        # self._parse_observations()

    # def parse_observations(self) -> Dict[str, xr.Dataset]:
    #    observations_by_type = {}
    #
    #    errors = []
    #    for obs_config in observation_configs.values():
    #        try:
    #            response_config = next(
    #                rc
    #                for rc in response_configs.values()
    #                if rc.response_type() == obs_config.response_type
    #                or (rc.name == obs_config.response_name and rc.name is not None)
    #            )
    #        except StopIteration:
    #            errors.append(
    #                ErrorInfo(
    #                    "Could not match observation to a response type or name"
    #                ).set_context_list(obs_config.line_from_ert_config)
    #            )
    #
    #        observation_ds = response_config.parse_observation_from_config(obs_config)
    #
    #        if observation_ds:
    #            if obs_config.response_type not in observations_by_type:
    #                observations_by_type[obs_config.response_type] = []
    #
    #            observations_by_type[obs_config.response_type].append(observation_ds)
    #
    #    # Merge by type & primary key
    #    return {
    #        obs_type: xr.concat(obs_ds_list, dim="obs_name")
    #        for obs_type, obs_ds_list in observations_by_type.items()
    #    }

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
    def from_dict(
        cls,
        config_dict: ConfigDict,
        response_types: Dict[str, ResponseConfigWithLifecycleHooks],
    ) -> EnsembleConfig:
        grid_file_path = config_dict.get(ConfigKeys.GRID)
        refcase_file_path = config_dict.get(ConfigKeys.REFCASE)
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
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

        eclbase = config_dict.get("ECLBASE")
        if eclbase is not None:
            eclbase = eclbase.replace("%d", "<IENS>")
        refcase_keys = []
        time_map = []
        data = None
        if refcase_file_path is not None:
            try:
                start_date, refcase_keys, time_map, data = read_summary(
                    refcase_file_path, ["*"]
                )
            except Exception as err:
                raise ConfigValidationError(f"Could not read refcase: {err}") from err

        observation_configs: Dict[str, ObservationConfig] = {}
        response_configs: Dict[str, ResponseConfigWithLifecycleHooks] = {}
        for response_type_cls in response_types.values():
            observation_keywords = response_type_cls.ert_config_observation_keyword()
            response_keywords = response_type_cls.ert_config_response_keyword()

            if not isinstance(observation_keywords, list):
                observation_keywords = [observation_keywords]

            if not isinstance(response_keywords, list):
                response_keywords = [response_keywords]

            # Find all occurrences in config

            for resp_kw in response_keywords:
                if resp_kw not in config_dict:
                    continue

                response_config_instances = response_type_cls.from_config_list(
                    config_dict[resp_kw]
                )
                for inst in response_config_instances:
                    response_configs[inst.name] = inst

            for obs_kw in observation_keywords:
                if obs_kw not in config_dict:
                    continue

                observations = config_dict[obs_kw]

                for line_from_ert_config in observations:
                    # We cannot validate the name&type against observations until we
                    # receive it from the forward model and read in the names, unless
                    # the response was specifically specified with a name
                    obs_config = ObservationConfig(
                        line_from_ert_config,
                        response_type_cls.response_type(),
                    )
                    observation_configs[obs_config.obs_name] = obs_config

        return cls(
            observation_configs=observation_configs,
            response_configs=response_configs,
            grid_file=grid_file_path,
            gendata_list=[GenDataConfig.from_config_list(g) for g in gen_data_list],
            genkw_list=[GenKwConfig.from_config_list(g) for g in gen_kw_list],
            surface_list=[SurfaceConfig.from_config_list(s) for s in surface_list],
            eclbase=eclbase,
            field_list=[make_field(f) for f in field_list],
            refcase=(
                Refcase(start_date, refcase_keys, time_map, data)
                if data is not None
                else None
            ),
        )

    def _node_info(self, object_type: Type[Any]) -> str:
        key_list = self.getKeylistFromImplType(object_type)
        return (
            f"{str(object_type).lower() + '_list'}="
            f"{[self[key] for key in key_list]}, "
        )

    def __repr__(self) -> str:
        return (
            "EnsembleConfig("
            + self._node_info(GenDataConfig)
            + self._node_info(GenKwConfig)
            + self._node_info(SurfaceConfig)
            + self._node_info(SummaryConfig)
            + self._node_info(Field)
            + f"grid_file={self._grid_file},"
            + f"refcase={self.refcase},"
            + ")"
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

    def check_unique_node(self, key: str) -> None:
        if key in self:
            raise ConfigValidationError(
                f"Config node with key {key!r} already present in ensemble config"
            )

    def addNode(self, config_node: Union[ParameterConfig, ResponseConfig]) -> None:
        assert config_node is not None
        self.check_unique_node(config_node.name)
        if isinstance(config_node, ParameterConfig):
            logger.info(
                f"Adding {type(config_node).__name__} config (of size {len(config_node)}) to parameter_configs"
            )
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
        return list(self.response_configs.keys())

    @property
    def keys(self) -> List[str]:
        return self.parameters + self.responses

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnsembleConfig):
            return False

        return (
            self.keys == other.keys
            and self._grid_file == other._grid_file
            and self.parameter_configs == other.parameter_configs
            and self.response_configs == other.response_configs
            and self.refcase == other.refcase
        )

    def get_summary_keys(self) -> List[str]:
        if "summary" in self:
            summary = self["summary"]
            if isinstance(summary, SummaryConfig):
                return sorted(set(summary.keys))
        return []

    @property
    def parameter_configuration(self) -> List[ParameterConfig]:
        return list(self.parameter_configs.values())

    @property
    def response_configuration(self) -> List[ResponseConfig]:
        return list(self.response_configs.values())
