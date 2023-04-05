import logging
import os
from typing import TYPE_CHECKING, Dict, Final, List, Tuple, Union

from cwrap import BaseCClass
from ecl.util.util import IntVector, StringList

from ert._c_wrappers import ResPrototype
from ert.parsing import ConfigValidationError
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.enums import (
    EnkfTruncationType,
    ErtImplType,
    FieldFileFormatType,
    LoadFailTypeEnum,
)

from .ext_param_config import ExtParamConfig
from .field_config import FieldConfig
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig
from .summary_config import SummaryConfig
from .surface_config import SurfaceConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


FIELD_FUNCTION_NAMES: Final = [
    "LN",
    "LOG10",
    "EXP0",
    "LOG",
    "EXP",
    "TRUNC_POW10",
    "LN0",
    "POW10",
]


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
    _alloc_field_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_field(char*, \
                                                           ecl_grid)",
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

    _alloc_gen_kw_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_KW_full(char*, \
                                                                 char*, \
                                                                 char*, \
                                                                 char*)",
        bind=False,
    )

    _alloc_surface_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_SURFACE_full(char*, \
                                                                  char*)",
        bind=False,
    )

    _update_field = ResPrototype(
        "void enkf_config_node_update_field(enkf_config_node, \
                                                      field_file_format_type_enum, \
                                                      enkf_truncation_type_enum, \
                                                      double, \
                                                      double, \
                                                      char*, \
                                                      char*, \
                                                      char*, \
                                                      char*)",
        bind=True,
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getImplementationType(self) -> ErtImplType:
        return self._get_impl_type()

    def getPointerReference(self):
        return self._get_ref()

    def getFieldModelConfig(self) -> FieldConfig:
        return FieldConfig.createCReference(self._get_ref(), parent=self)

    def getSurfaceModelConfig(self) -> SurfaceConfig:
        return SurfaceConfig.createCReference(self._get_ref(), parent=self)

    def getDataModelConfig(self) -> GenDataConfig:
        return GenDataConfig.createCReference(self._get_ref(), parent=self)

    def getKeywordModelConfig(self) -> GenKwConfig:
        return GenKwConfig.createCReference(self._get_ref(), parent=self)

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

    # GEN KW FULL creation
    @classmethod
    def create_gen_kw(
        cls,
        key,
        template_file,
        parameter_file,
        gen_kw_format,
    ):
        config_node = cls._alloc_gen_kw_full(
            key,
            gen_kw_format,
            template_file,
            parameter_file,
        )

        return config_node

    # SURFACE FULL creation
    @classmethod
    def create_surface(
        cls,
        key,
        init_file_fmt,
        output_file,
        base_surface_file,
        forward_init=False,
    ):
        msg = f"Following issues found for SURFACE node {key!r}:\n"
        valid = True
        if init_file_fmt is None:
            msg = msg + f"Missing {ConfigKeys.INIT_FILES}:/path/to/input/files\n"
            valid = False
        elif (
            not forward_init
            and "%d" not in init_file_fmt
            and not os.path.exists(os.path.realpath(init_file_fmt))
        ):
            msg += f"{ConfigKeys.INIT_FILES}:{init_file_fmt!r} File not found "
            valid = False
        if output_file is None:
            msg = msg + "Missing OUTPUT_FILE:/path/to/output_file\n"
            valid = False
        if base_surface_file is None:
            valid = False
            msg = (
                msg
                + f"Missing {ConfigKeys.BASE_SURFACE_KEY}:/path/to/base_surface_file\n"
            )
        elif not os.path.exists(os.path.realpath(base_surface_file)):
            msg += (
                f"{ConfigKeys.BASE_SURFACE_KEY}:{base_surface_file!r} File not found "
            )
            valid = False
        if not valid:
            logger.error(msg)
            raise ConfigValidationError(msg)

        base_surface_file = os.path.realpath(base_surface_file)
        config_node = cls._alloc_surface_full(
            key,
            base_surface_file,
        )

        return config_node

    # FIELD FULL creation
    @classmethod
    def create_field(
        cls,
        key,
        var_type_string,
        grid,
        ecl_file,
        init_transform,
        output_transform,
        input_transform,
        min_key,
        max_key,
    ):
        # pylint: disable=unsupported-binary-operation
        # (false positive from the cwrap class BaseCEnum)
        truncation = EnkfTruncationType.TRUNCATE_NONE
        value_min = -1
        value_max = -1
        if min_key is not None:
            if isinstance(min_key, str):
                min_key = float(min_key)
            value_min = min_key
            truncation = truncation | EnkfTruncationType.TRUNCATE_MIN
        if max_key is not None:
            if isinstance(max_key, str):
                max_key = float(max_key)
            value_max = max_key
            truncation = truncation | EnkfTruncationType.TRUNCATE_MAX

        config_node = cls._alloc_field_node(key, grid)

        cls.verifyValidFieldName(init_transform, "INIT_TRANSFORM")
        cls.verifyValidFieldName(input_transform, "INPUT_TRANSFORM")
        cls.verifyValidFieldName(output_transform, "OUTPUT_TRANSFORM")

        file_format_enum = (
            FieldFileFormatType.ECL_KW_FILE_ALL_CELLS
            if ecl_file
            else FieldFileFormatType.FILE_FORMAT_NULL
        )

        if ecl_file.upper().endswith("GRDECL"):
            file_format_enum = FieldFileFormatType.ECL_GRDECL_FILE
        elif ecl_file.upper().endswith("ROFF"):
            file_format_enum = FieldFileFormatType.RMS_ROFF_FILE

        if var_type_string == ConfigKeys.PARAMETER_KEY:
            config_node._update_field(
                file_format_enum,
                truncation,
                value_min,
                value_max,
                init_transform,
                input_transform,
                output_transform,
                ecl_file,
            )

        return config_node

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(
            f"key = {self.getKey()}, "
            f"implementation = {self.getImplementationType()}"
        )

    def verifyValidFieldName(func_name: str, field_name: str):
        if func_name and func_name not in FIELD_FUNCTION_NAMES:
            raise ValueError(f"FIELD {field_name}:{func_name} is an invalid function")

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
                self.getImplementationType() == ErtImplType.GEN_KW
                and self.getKeywordModelConfig() != other.getKeywordModelConfig(),
                self.getImplementationType() == ErtImplType.SUMMARY
                and self.getSummaryModelConfig() != other.getSummaryModelConfig(),
                self.getImplementationType() == ErtImplType.FIELD
                and self.getFieldModelConfig() != other.getFieldModelConfig(),
            ]
        ):
            return False

        return True
