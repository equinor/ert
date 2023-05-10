import logging
from typing import Dict, List, Tuple, Union

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import ErtImplType

from .ext_param_config import ExtParamConfig
from .summary_config import SummaryConfig

logger = logging.getLogger(__name__)


class EnkfConfigNode(BaseCClass):
    TYPE_NAME = "enkf_config_node"

    _alloc = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc(ert_impl_type_enum, \
                                                     char*, \
                                                     void*)",
        bind=False,
    )
    _alloc_summary_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_summary(char*)",
        bind=False,
    )
    _get_ref = ResPrototype(
        "void* enkf_config_node_get_ref(enkf_config_node)"
    )  # todo: fix return type
    _get_impl_type = ResPrototype(
        "ert_impl_type_enum enkf_config_node_get_impl_type(enkf_config_node)"
    )
    _get_key = ResPrototype("char* enkf_config_node_get_key(enkf_config_node)")
    _free = ResPrototype("void enkf_config_node_free(enkf_config_node)")

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getImplementationType(self) -> ErtImplType:
        return self._get_impl_type()

    def getPointerReference(self):
        return self._get_ref()

    def getSummaryModelConfig(self) -> SummaryConfig:
        return SummaryConfig.createCReference(self._get_ref(), parent=self)

    @classmethod
    def createSummaryConfigNode(cls, key: str) -> "EnkfConfigNode":
        return cls._alloc_summary_node(key)

    @classmethod
    def create_ext_param(
        cls,
        key: str,
        input_keys: Union[List[str], Dict[str, List[Tuple[str, str]]]],
    ) -> "EnkfConfigNode":
        config = ExtParamConfig(key, input_keys)
        node = cls._alloc(
            ErtImplType.EXT_PARAM,
            key,
            ExtParamConfig.from_param(config),
        )
        config.convertToCReference(node)  # config gets freed when node dies
        return node

    @staticmethod
    def validate_gen_data_format(formatted_file: str = "") -> bool:
        return formatted_file.count("%d") == 1

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(
            f"key = {self.getKey()}, "
            f"implementation = {self.getImplementationType()}"
        )

    def getModelConfig(
        self,
    ) -> Union[SummaryConfig, ExtParamConfig]:
        implementation_type = self.getImplementationType()

        if implementation_type == ErtImplType.SUMMARY:
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
                self.getImplementationType() == ErtImplType.SUMMARY
                and self.getSummaryModelConfig() != other.getSummaryModelConfig(),
            ]
        ):
            return False

        return True
