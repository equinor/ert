#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ensemble_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ecl.util.util import StringList
from ecl.grid import EclGrid
from ecl.summary import EclSum
from res import ResPrototype
from res.enkf import SummaryKeyMatcher
from res.config import ConfigContent
from res.enkf.config import EnkfConfigNode, CustomKWConfig
from res.enkf.enums import EnkfVarType, ErtImplType



class EnsembleConfig(BaseCClass):
    TYPE_NAME = "ens_config"
    _alloc = ResPrototype("void* ensemble_config_alloc(config_content, ecl_grid, ecl_sum)", bind=False)
    _free = ResPrototype("void ensemble_config_free( ens_config )")
    _has_key = ResPrototype("bool ensemble_config_has_key( ens_config , char* )")
    _size = ResPrototype("int ensemble_config_get_size( ens_config)")
    _get_node = ResPrototype("enkf_config_node_ref ensemble_config_get_node( ens_config , char*)")
    _alloc_keylist = ResPrototype("stringlist_obj ensemble_config_alloc_keylist( ens_config )")
    _add_summary = ResPrototype("enkf_config_node_ref ensemble_config_add_summary( ens_config, char*, int)")
    _add_gen_kw = ResPrototype("enkf_config_node_ref ensemble_config_add_gen_kw( ens_config, char*)")
    _add_field = ResPrototype("enkf_config_node_ref ensemble_config_add_field( ens_config, char*, ecl_grid)")
    _alloc_keylist_from_var_type = ResPrototype("stringlist_obj ensemble_config_alloc_keylist_from_var_type(ens_config, enkf_var_type_enum)")
    _alloc_keylist_from_impl_type = ResPrototype("stringlist_obj ensemble_config_alloc_keylist_from_impl_type(ens_config, ert_impl_type_enum)")
    _add_node = ResPrototype("void ensemble_config_add_node( ens_config , enkf_config_node )")
    _summary_key_matcher = ResPrototype("summary_key_matcher_ref ensemble_config_get_summary_key_matcher(ens_config)")
    _add_defined_custom_kw = ResPrototype("enkf_config_node_ref ensemble_config_add_defined_custom_kw(ens_config, char*, integer_hash)")



    def __init__(self, config_content=None, grid=None, refcase=None):
        c_ptr = self._alloc(config_content, grid, refcase)

        if c_ptr is None:
            raise ValueError("Failed to construct EnsembleConfig instance")

        super(EnsembleConfig , self).__init__(c_ptr)

    def __len__(self):
        return self._size( )

    def __getitem__(self , key):
        """ @rtype: EnkfConfigNode """
        if key in self:
            return self._get_node(key).setParent(self)
        else:
            raise KeyError("The key:%s is not in the ensemble configuration" % key)

    def getNode(self, key):
        return self[key]

    def alloc_keylist(self):
        """ @rtype: StringList """
        return self._alloc_keylist( )

    def add_summary(self, key):
        """ @rtype: EnkfConfigNode """
        return self._add_summary(key, 2).setParent(self)

    def add_gen_kw(self, key):
        """ @rtype: EnkfConfigNode """
        return self._add_gen_kw(key).setParent(self)


    def addNode(self , config_node):
        assert isinstance(config_node , EnkfConfigNode)
        self._add_node( config_node )
        config_node.convertToCReference( self )


    def add_field(self, key, eclipse_grid):
        """ @rtype: EnkfConfigNode """
        return self._add_field(key, eclipse_grid).setParent(self)

    def getKeylistFromVarType(self, var_mask):
        """ @rtype: StringList """
        assert isinstance(var_mask, EnkfVarType)
        return self._alloc_keylist_from_var_type(var_mask)

    def getKeylistFromImplType(self, ert_impl_type):
        """ @rtype: StringList """
        assert isinstance(ert_impl_type, ErtImplType)
        return self._alloc_keylist_from_impl_type(ert_impl_type)

    def __contains__(self, key):
        return self._has_key( key)

    def getSummaryKeyMatcher(self):
        """ @rtype: SummaryKeyMatcher """
        return self._summary_key_matcher( )

    def free(self):
        self._free( )

    def addDefinedCustomKW(self, group_name, definition):
        """ @rtype: EnkfConfigNode """
        if not group_name in self:
            type_hash = CustomKWConfig.convertDefinition(definition)
            self._add_defined_custom_kw(group_name, type_hash)

        return self[group_name]

