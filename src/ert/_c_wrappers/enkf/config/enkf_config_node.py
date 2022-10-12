import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from cwrap import BaseCClass
from ecl.grid import EclGrid
from ecl.util.util import IntVector, StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.enums import (
    EnkfTruncationType,
    EnkfVarType,
    ErtImplType,
    LoadFailTypeEnum,
)

from .ext_param_config import ExtParamConfig
from .field_config import FieldConfig
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig
from .summary_config import SummaryConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


class EnkfConfigNode(BaseCClass):
    TYPE_NAME = "enkf_config_node"

    _alloc = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc(enkf_var_type_enum, \
                                                     ert_impl_type_enum, \
                                                     bool, \
                                                     char*, \
                                                     char*, \
                                                     char*, \
                                                     char*, \
                                                     void*)",
        bind=False,
    )
    _alloc_gen_data_everest = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_everest(char*, \
                                                                      char*, \
                                                                      int_vector)",
        bind=False,
    )
    _alloc_summary_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_summary(char*, load_fail_type)",
        bind=False,
    )
    _alloc_field_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_field(char*, \
                                                           ecl_grid, \
                                                           void*, \
                                                           bool)",
        bind=False,
    )
    _get_ref = ResPrototype(
        "void* enkf_config_node_get_ref(enkf_config_node)"
    )  # todo: fix return type
    _get_impl_type = ResPrototype(
        "ert_impl_type_enum enkf_config_node_get_impl_type(enkf_config_node)"
    )
    _get_enkf_outfile = ResPrototype(
        "char* enkf_config_node_get_enkf_outfile(enkf_config_node)"
    )
    _get_enkf_infile = ResPrototype(
        "char* enkf_config_node_get_enkf_infile(enkf_config_node)"
    )
    _get_init_file_fmt = ResPrototype(
        "char* enkf_config_node_get_init_file_fmt(enkf_config_node)"
    )
    _get_var_type = ResPrototype(
        "enkf_var_type_enum enkf_config_node_get_var_type(enkf_config_node)"
    )  # todo: fix return type as enum
    _get_key = ResPrototype("char* enkf_config_node_get_key(enkf_config_node)")
    _get_obs_keys = ResPrototype(
        "stringlist_ref enkf_config_node_get_obs_keys(enkf_config_node)"
    )
    _free = ResPrototype("void enkf_config_node_free(enkf_config_node)")
    _use_forward_init = ResPrototype(
        "bool enkf_config_node_use_forward_init(enkf_config_node)"
    )

    _alloc_gen_data_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_full(char*, \
                                                                   char*, \
                                                                   gen_data_file_format_type, \
                                                                   int_vector)",  # noqa
        bind=False,
    )

    _alloc_gen_kw_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_KW_full(char*, \
                                                                 bool, \
                                                                 char*, \
                                                                 char*, \
                                                                 char*, \
                                                                 char*, \
                                                                 char*)",
        bind=False,
    )

    _alloc_surface_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_SURFACE_full(char*, \
                                                                  bool, \
                                                                  char*, \
                                                                  char*, \
                                                                  char*)",
        bind=False,
    )

    _update_parameter_field = ResPrototype(
        "void enkf_config_node_update_parameter_field(enkf_config_node, \
                                                      char*, \
                                                      char*, \
                                                      enkf_truncation_type_enum, \
                                                      double, \
                                                      double, \
                                                      char*, \
                                                      char*)",
        bind=True,
    )
    _update_general_field = ResPrototype(
        "void enkf_config_node_update_general_field(enkf_config_node, \
                                                    char*, \
                                                    char*, \
                                                    char*, \
                                                    enkf_truncation_type_enum, \
                                                    double, \
                                                    double, \
                                                    char*, \
                                                    char*, \
                                                    char*)",
        bind=True,
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getImplementationType(self) -> ErtImplType:
        return self._get_impl_type()

    def getVariableType(self) -> EnkfVarType:
        return self._get_var_type()

    def getPointerReference(self):
        return self._get_ref()

    def getUseForwardInit(self) -> bool:
        return self._use_forward_init()

    def get_enkf_outfile(self) -> str:
        return self._get_enkf_outfile()

    def getFieldModelConfig(self) -> FieldConfig:
        return FieldConfig.createCReference(self._get_ref(), parent=self)

    def getDataModelConfig(self) -> GenDataConfig:
        return GenDataConfig.createCReference(self._get_ref(), parent=self)

    def getKeywordModelConfig(self) -> GenKwConfig:
        return GenKwConfig.createCReference(self._get_ref(), parent=self)

    def getSummaryModelConfig(self) -> SummaryConfig:
        return SummaryConfig.createCReference(self._get_ref(), parent=self)

    def get_enkf_infile(self):
        return self._get_enkf_infile()

    def get_init_file_fmt(self) -> str:
        return self._get_init_file_fmt()

    def getObservationKeys(self) -> StringList:
        return self._get_obs_keys().setParent(self)

    @classmethod
    def createSummaryConfigNode(
        cls, key: str, load_fail_type: LoadFailTypeEnum
    ) -> "EnkfConfigNode":
        assert isinstance(load_fail_type, LoadFailTypeEnum)
        return cls._alloc_summary_node(key, load_fail_type)

    @classmethod
    def createFieldConfigNode(
        cls, key, grid: EclGrid, trans_table=None, forward_init=False
    ) -> "EnkfConfigNode":
        return cls._alloc_field_node(key, grid, trans_table, forward_init)

    @classmethod
    def create_ext_param(
        cls,
        key: str,
        input_keys: Union[List[str], Dict[str, List[Tuple[str, str]]]],
        output_file: Optional[str] = None,
    ) -> ExtParamConfig:
        config = ExtParamConfig(key, input_keys)
        output_file = output_file or key + ".json"
        node = cls._alloc(
            EnkfVarType.EXT_PARAMETER,
            ErtImplType.EXT_PARAM,
            False,
            key,
            None,
            output_file,
            None,
            ExtParamConfig.from_param(config),
        )
        config.convertToCReference(node)  # config gets freed when node dies
        return node

    # This method only exposes the details relevant for Everest usage.
    @classmethod
    def create_gen_data(
        cls, key: str, file_fmt: str, report_steps: "Iterable[int]" = (0,)
    ):
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_everest(key, file_fmt, active_steps)
        if config_node is None:
            raise ValueError(f"Failed to create GEN_DATA node for:{key}")

        return config_node

    # GEN DATA FULL creation
    @classmethod
    def create_gen_data_full(
        cls,
        key,
        result_file,
        input_format,
        report_steps,
    ):
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_full(
            key,
            result_file,
            input_format,
            active_steps,
        )
        if config_node is None:
            raise ValueError(
                f"Failed to create GEN_DATA with FULL specs node for:{key}"
            )

        return config_node

    # GEN KW FULL creation
    @classmethod
    def create_gen_kw(
        cls,
        key,
        template_file,
        enkf_outfile,
        parameter_file,
        forward_init,
        init_file_fmt,
        gen_kw_format,
    ):

        config_node = cls._alloc_gen_kw_full(
            key,
            forward_init,
            gen_kw_format,
            template_file,
            enkf_outfile,
            parameter_file,
            init_file_fmt,
        )
        if config_node is None:
            raise ValueError(f"Failed to create GEN KW node for:{key}")

        return config_node

    # SURFACE FULL creation
    @classmethod
    def create_surface(
        cls,
        key,
        init_file_fmt,
        output_file,
        base_surface_file,
        forward_init,
    ):

        if base_surface_file is not None:
            base_surface_file = os.path.realpath(base_surface_file)
        config_node = cls._alloc_surface_full(
            key,
            forward_init,
            output_file,
            base_surface_file,
            init_file_fmt,
        )
        if config_node is None:
            raise ValueError(f"Failed to create SURFACE node for:{key}")

        return config_node

    # FIELD FULL creation
    @classmethod
    def create_field(
        cls,
        key,
        var_type_string,
        grid,
        field_trans_table,
        ecl_file,
        enkf_infile,
        forward_init,
        init_transform,
        output_transform,
        input_transform,
        min_key,
        max_key,
        init_file_fmt,
    ):

        # pylint: disable=unsupported-binary-operation
        # (false positive from the cwrap class BaseCEnum)
        truncation = EnkfTruncationType.TRUNCATE_NONE
        value_min = -1
        value_max = -1
        if min_key is not None:
            value_min = min_key
            truncation = truncation | EnkfTruncationType.TRUNCATE_MIN
        if max_key is not None:
            value_max = max_key
            truncation = truncation | EnkfTruncationType.TRUNCATE_MAX

        config_node = cls._alloc_field_node(key, grid, field_trans_table, forward_init)
        if config_node is None:
            raise ValueError(f"Failed to create FIELD node for:{key}")

        if var_type_string == ConfigKeys.PARAMETER_KEY:
            config_node._update_parameter_field(
                ecl_file,
                init_file_fmt,
                truncation,
                value_min,
                value_max,
                init_transform,
                output_transform,
            )

        elif var_type_string == ConfigKeys.GENERAL_KEY:
            config_node._update_general_field(
                ecl_file,
                enkf_infile,
                init_file_fmt,
                truncation,
                value_min,
                value_max,
                init_transform,
                input_transform,
                output_transform,
            )

        return config_node

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(
            f"key = {self.getKey()}, var_type = {self.getVariableType()}, "
            f"implementation = {self.getImplementationType()}"
        )

    def getModelConfig(
        self,
    ) -> Union[FieldConfig, GenDataConfig, GenKwConfig, SummaryConfig, ExtParamConfig]:
        implementation_type = self.getImplementationType()

        if implementation_type == ErtImplType.FIELD:
            return self.getFieldModelConfig()
        elif implementation_type == ErtImplType.GEN_DATA:
            return self.getDataModelConfig()
        elif implementation_type == ErtImplType.GEN_KW:
            return self.getKeywordModelConfig()
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

    def getKey(self):
        return self._get_key()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other) -> bool:
        if self.getImplementationType() != other.getImplementationType():
            return False

        if self.getKey() != other.getKey():
            return False

        if self.getImplementationType() == ErtImplType.EXT_PARAM:
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
        elif self.getImplementationType() == ErtImplType.GEN_DATA:
            if self.getDataModelConfig() != other.getDataModelConfig():
                return False
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False
            if self.get_enkf_infile() != other.get_enkf_infile():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
        elif self.getImplementationType() == ErtImplType.GEN_KW:
            if self.getKeywordModelConfig() != other.getKeywordModelConfig():
                return False
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
        elif self.getImplementationType() == ErtImplType.SUMMARY:
            if self.getSummaryModelConfig() != other.getSummaryModelConfig():
                return False
        elif self.getImplementationType() == ErtImplType.SURFACE:
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False
        elif self.getImplementationType() == ErtImplType.FIELD:
            if self.getFieldModelConfig() != other.getFieldModelConfig():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False

        return True
