#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'enkf_config_node.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from cwrap import BaseCClass

from ecl.grid import EclGrid
from ecl.util.util import StringList, IntVector

from res import ResPrototype
from res.enkf.config import (
    FieldConfig,
    GenDataConfig,
    GenKwConfig,
    SummaryConfig,
    ExtParamConfig,
)
from res.enkf.enums import (
    EnkfTruncationType,
    ErtImplType,
    LoadFailTypeEnum,
    EnkfVarType,
)
from res.enkf import ConfigKeys
import os


class EnkfConfigNode(BaseCClass):
    TYPE_NAME = "enkf_config_node"

    _alloc = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc(enkf_var_type_enum, ert_impl_type_enum, bool, char*, char* , char*, char*, void*)",
        bind=False,
    )
    _alloc_gen_data_everest = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_everest(char*, char* , int_vector)",
        bind=False,
    )
    _alloc_summary_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_summary(char*, load_fail_type)",
        bind=False,
    )
    _alloc_field_node = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_field(char*, ecl_grid, void*, bool)",
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
    _get_min_std_file = ResPrototype(
        "char* enkf_config_node_get_min_std_file(enkf_config_node)"
    )
    _get_enkf_infile = ResPrototype(
        "char* enkf_config_node_get_enkf_infile(enkf_config_node)"
    )
    _get_init_file = ResPrototype(
        "char* enkf_config_node_get_FIELD_fill_file(enkf_config_node, path_fmt)"
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

    # ensemble config aux
    _alloc_gen_param_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_PARAM_full( char*,\
                                                                                                      bool, \
                                                                                                      gen_data_file_format_type, \
                                                                                                      gen_data_file_format_type, \
                                                                                                      char*, \
                                                                                                      char*, \
                                                                                                      char*, \
                                                                                                      char*, \
                                                                                                      char*)",
        bind=False,
    )

    _alloc_gen_data_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_DATA_full( char*,\
                                                                                                    char*, \
                                                                                                    gen_data_file_format_type, \
                                                                                                    int_vector, \
                                                                                                    char*, \
                                                                                                    char*, \
                                                                                                    char*, \
                                                                                                    char*)",
        bind=False,
    )

    _alloc_gen_kw_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_GEN_KW_full( char*,\
                                                                                                bool, \
                                                                                                char*, \
                                                                                                char*, \
                                                                                                char*, \
                                                                                                char*, \
                                                                                                char*, \
                                                                                                char*)",
        bind=False,
    )

    _alloc_surface_full = ResPrototype(
        "enkf_config_node_obj enkf_config_node_alloc_SURFACE_full( char*,\
                                                                                                  bool, \
                                                                                                  char*, \
                                                                                                  char*, \
                                                                                                  char*, \
                                                                                                  char*)",
        bind=False,
    )

    _alloc_container = ResPrototype(
        "enkf_config_node_obj enkf_config_node_new_container(char*)", bind=False
    )
    _update_container = ResPrototype(
        "void enkf_config_node_update_container(enkf_config_node, enkf_config_node)"
    )
    _get_container_size = ResPrototype(
        "int enkf_config_node_container_size(enkf_config_node)"
    )
    _iget_container_key = ResPrototype(
        "char* enkf_config_node_iget_container_key(enkf_config_node, int)"
    )
    _update_parameter_field = ResPrototype(
        "void enkf_config_node_update_parameter_field(enkf_config_node, \
                                                                                         char*, \
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

    def get_container_size(self):
        return self._get_container_size()

    def get_container_key(self, index):
        return self._iget_container_key(index)

    def getImplementationType(self):
        """ @rtype: ErtImplType """
        return self._get_impl_type()

    def getVariableType(self):
        return self._get_var_type()

    def getPointerReference(self):
        return self._get_ref()

    def getUseForwardInit(self):
        return self._use_forward_init()

    def getInitFile(self, model_config):
        return self._enkf_config_node_get_init_file(model_config.getRunpathFormat())

    def get_min_std_file(self):
        return self._get_min_std_file()

    def get_enkf_outfile(self):
        return self._get_enkf_outfile()

    def getFieldModelConfig(self):
        """ @rtype: FieldConfig """
        return FieldConfig.createCReference(self._get_ref(), parent=self)

    def getDataModelConfig(self):
        """ @rtype: GenDataConfig """
        return GenDataConfig.createCReference(self._get_ref(), parent=self)

    def getKeywordModelConfig(self):
        """ @rtype: GenKWConfig """
        return GenKwConfig.createCReference(self._get_ref(), parent=self)

    def getSummaryModelConfig(self):
        """ @rtype: SummaryConfig """
        return SummaryConfig.createCReference(self._get_ref(), parent=self)

    def get_enkf_infile(self):
        return self._get_enkf_infile()

    def get_init_file_fmt(self):
        return self._get_init_file_fmt()

    def getObservationKeys(self):
        """ @rtype:  StringList """
        return self._get_obs_keys().setParent(self)

    @classmethod
    def createSummaryConfigNode(cls, key, load_fail_type):
        """
         @type key: str
         @type load_fail_type: LoadFailTypeEnum
        @rtype: EnkfConfigNode
        """

        assert isinstance(load_fail_type, LoadFailTypeEnum)
        return cls._alloc_summary_node(key, load_fail_type)

    @classmethod
    def createFieldConfigNode(cls, key, grid, trans_table=None, forward_init=False):
        """
        @type grid: EclGrid
        @rtype: EnkfConfigNode
        """
        return cls._alloc_field_node(key, grid, trans_table, forward_init)

    @classmethod
    def create_ext_param(cls, key, input_keys, output_file=None):
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
    def create_gen_data(cls, key, file_fmt, report_steps=(0,)):
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_everest(key, file_fmt, active_steps)
        if config_node is None:
            raise ValueError("Failed to create GEN_DATA node for:%s" % key)

        return config_node

    # GEN DATA FULL creation
    @classmethod
    def create_gen_data_full(
        cls,
        key,
        result_file,
        input_format,
        report_steps,
        ecl_file,
        init_file_fmt,
        template_file,
        data_key,
    ):
        active_steps = IntVector()
        for step in report_steps:
            active_steps.append(step)

        config_node = cls._alloc_gen_data_full(
            key,
            result_file,
            input_format,
            active_steps,
            ecl_file,
            init_file_fmt,
            template_file,
            data_key,
        )
        if config_node is None:
            raise ValueError(
                "Failed to create GEN_DATA with FULL specs node for:%s" % key
            )

        return config_node

    # GEN PARAM FULL creation
    @classmethod
    def create_gen_param(
        cls,
        key,
        forward_init,
        input_format,
        output_format,
        init_file_fmt,
        ecl_file,
        min_std_file,
        template_file,
        data_key,
    ):

        config_node = cls._alloc_gen_param_full(
            key,
            forward_init,
            input_format,
            output_format,
            init_file_fmt,
            ecl_file,
            min_std_file,
            template_file,
            data_key,
        )
        if config_node is None:
            raise ValueError("Failed to create GEN_PARAM node for:%s" % key)

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
        mid_std_file,
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
            mid_std_file,
            init_file_fmt,
        )
        if config_node is None:
            raise ValueError("Failed to create GEN KW node for:%s" % key)

        return config_node

    # SURFACE FULL creation
    @classmethod
    def create_surface(
        cls,
        key,
        init_file_fmt,
        output_file,
        base_surface_file,
        min_std_file,
        forward_init,
    ):

        if base_surface_file is not None:
            base_surface_file = os.path.realpath(base_surface_file)
        config_node = cls._alloc_surface_full(
            key,
            forward_init,
            output_file,
            base_surface_file,
            min_std_file,
            init_file_fmt,
        )
        if config_node is None:
            raise ValueError("Failed to create SURFACE node for:%s" % key)

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
        min_std_file,
        min_key,
        max_key,
        init_file_fmt,
    ):

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
            raise ValueError("Failed to create FIELD node for:%s" % key)

        if var_type_string == ConfigKeys.PARAMETER_KEY:
            config_node._update_parameter_field(
                ecl_file,
                init_file_fmt,
                min_std_file,
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
                min_std_file,
                truncation,
                value_min,
                value_max,
                init_transform,
                input_transform,
                output_transform,
            )

        return config_node

    # CONTAINER creation
    @classmethod
    def create_container(cls, key):
        config_node = cls._alloc_container(key)

        if config_node is None:
            raise ValueError("Failed to create CONTAINER node for:%s" % key)

        return config_node

    def free(self):
        self._free()

    def __repr__(self):
        key = self.getKey()
        vt = self.getVariableType()
        imp = self.getImplementationType()
        content = "key = %s, var_type = %s, implementation = %s" % (key, vt, imp)
        return self._create_repr(content)

    def getModelConfig(self):
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
            print(
                "[EnkfConfigNode::getModelConfig()] Unhandled implementation model type: %i"
                % implementation_type
            )
            # raise NotImplementedError("Unknown model type: %i" % type)

    def getKey(self):
        return self._get_key()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        """ @rtype: bool"""
        if self.getImplementationType() != other.getImplementationType():
            return False

        if self.getKey() != other.getKey():
            return False

        if self.getImplementationType() == ErtImplType.EXT_PARAM:
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_min_std_file() != other.get_min_std_file():
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
            if self.get_min_std_file() != other.get_min_std_file():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
        elif self.getImplementationType() == ErtImplType.CONTAINER:
            a = [self.get_container_key(i) for i in range(self.get_container_size())]
            b = [other.get_container_key(i) for i in range(other.get_container_size())]
            if a != b:
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
            if self.get_min_std_file() != other.get_min_std_file():
                return False
        elif self.getImplementationType() == ErtImplType.FIELD:
            if self.getFieldModelConfig() != other.getFieldModelConfig():
                return False
            if self.getUseForwardInit() != other.getUseForwardInit():
                return False
            if self.get_init_file_fmt() != other.get_init_file_fmt():
                return False
            if self.get_min_std_file() != other.get_min_std_file():
                return False
            if self.get_enkf_outfile() != other.get_enkf_outfile():
                return False

        return True
