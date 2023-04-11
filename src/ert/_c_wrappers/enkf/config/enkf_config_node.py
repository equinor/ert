import logging
import os
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from cwrap import BaseCClass
from ecl.util.util import IntVector, StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import ErtImplType, LoadFailTypeEnum

from .ext_param_config import ExtParamConfig
from .gen_data_config import GenDataConfig
from .summary_config import SummaryConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class EnkfConfigNode(BaseCClass):
    TYPE_NAME = "enkf_config_node"

    _alloc = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc(ert_impl_type_enum, \
                                                     char*, \
                                                     void*)",
        bind=False,
    )
    _alloc_gen_data_everest = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_everest(char*, \
                                                                      int_vector)",
        bind=False,
    )
    _alloc_summary_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_summary(char*, load_fail_type)",
        bind=False,
    )
    _get_ref = ResPrototype(
        "void* enkf_config_node_get_ref(enkf_config_node)"
    )  # todo: fix return type
    _get_impl_type = ResPrototype(
        "ert_impl_type_enum enkf_config_node_get_impl_type(enkf_config_node)"
    )
    _get_key = ResPrototype("char* enkf_config_node_get_key(enkf_config_node)")
    _get_obs_keys = ResPrototype(
        "stringlist_ref enkf_config_node_get_obs_keys(enkf_config_node)"
    )
    _free = ResPrototype("void enkf_config_node_free(enkf_config_node)")

    _alloc_gen_data_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_full(char*, \
                                                                   gen_data_file_format_type, \
                                                                   int_vector)",  # noqa
        bind=False,
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getImplementationType(self) -> ErtImplType:
        return self._get_impl_type()

    def getPointerReference(self):
        return self._get_ref()

    def getDataModelConfig(self) -> GenDataConfig:
        return GenDataConfig.createCReference(self._get_ref(), parent=self)

    def getSummaryModelConfig(self) -> SummaryConfig:
        return SummaryConfig.createCReference(self._get_ref(), parent=self)

    def getObservationKeys(self) -> StringList:
        return self._get_obs_keys().setParent(self)

    @classmethod
    def createSummaryConfigNode(
        cls, key: str, load_fail_type: LoadFailTypeEnum
    ) -> "EnkfConfigNode":
        assert isinstance(load_fail_type, LoadFailTypeEnum)
        return cls._alloc_summary_node(key, load_fail_type)

    @classmethod
    def create_ext_param(
        cls,
        key: str,
        input_keys: Union[List[str], Dict[str, List[Tuple[str, str]]]],
    ) -> ExtParamConfig:
        config = ExtParamConfig(key, input_keys)
        node = cls._alloc(
            ErtImplType.EXT_PARAM,
            key,
            ExtParamConfig.from_param(config),
        )
        config.convertToCReference(node)  # config gets freed when node dies
        return node

    # This method only exposes the details relevant for Everest usage.
    @classmethod
    def create_gen_data(cls, key: str, report_steps: "Iterable[int]" = (0,)):
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_everest(key, active_steps)
        return config_node

    @staticmethod
    def validate_gen_data_format(formatted_file: str = "") -> bool:
        return formatted_file.count("%d") == 1

    # GEN DATA FULL creation
    @classmethod
    def create_gen_data_full(
        cls,
        key,
        result_file,
        input_format,
        report_steps,
    ):
        if os.path.isabs(result_file) or "%d" not in result_file:
            msg = (
                f"The RESULT_FILE:{result_file} setting for {key} is invalid - "
                "must have an embedded %d - and be a relative path"
            )
            logger.error(msg)
            return None
        if not report_steps:
            msg = (
                "The GEN_DATA keywords must have a REPORT_STEPS:xxxx defined"
                "Several report steps separated with ',' and ranges with '-'"
                "can be listed"
            )
            logger.error(msg)
            return None
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_full(
            key,
            input_format,
            active_steps,
        )

        return config_node

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(
            f"key = {self.getKey()}, "
            f"implementation = {self.getImplementationType()}"
        )

    def getModelConfig(
        self,
    ) -> Union[GenDataConfig, SummaryConfig, ExtParamConfig]:
        implementation_type = self.getImplementationType()

        if implementation_type == ErtImplType.GEN_DATA:
            return self.getDataModelConfig()
        elif implementation_type == ErtImplType.SUMMARY:
            return SummaryConfig.createCReference(
                self.getPointerReference(), parent=self
            )
        elif implementation_type == ErtImplType.EXT_PARAM:
            return ExtParamConfig.createCReference(
                self.getPointerReference(), parent=self
            )
        else:
            raise ValueError(
                "[EnkfConfigNode::getModelConfig()] "
                f"Unhandled implementation model type: {implementation_type:i}"
            )

    def getKey(self) -> str:
        return self._get_key()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other) -> bool:
        if any(
            (
                self.getImplementationType() != other.getImplementationType(),
                self.getKey() != other.getKey(),
            )
        ):
            return False

        if any(
            [
                self.getImplementationType() == ErtImplType.GEN_DATA
                and self.getDataModelConfig() != other.getDataModelConfig(),
                self.getImplementationType() == ErtImplType.SUMMARY
                and self.getSummaryModelConfig() != other.getSummaryModelConfig(),
            ]
        ):
            return False

        return True
