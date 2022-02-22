from typing import Dict, List
from cwrap import BaseCClass

from res import _lib
from res import ResPrototype
from res.enkf.local_obsdata import LocalObsdata
from res.enkf.local_obsdata_node import LocalObsdataNode
from res.enkf.row_scaling import RowScaling


class LocalMinistep(BaseCClass):
    TYPE_NAME = "local_ministep"

    _add_node = ResPrototype(
        "void local_ministep_add_obsdata_node(local_ministep, local_obsdata_node)"
    )
    _get_local_obs_data = ResPrototype(
        "local_obsdata_ref local_ministep_get_obsdata(local_ministep)"
    )
    _free = ResPrototype("void local_ministep_free(local_ministep)")
    _attach_obsdata = ResPrototype(
        "void local_ministep_add_obsdata(local_ministep, local_obsdata)"
    )
    _name = ResPrototype("char* local_ministep_get_name(local_ministep)")
    _data_size = ResPrototype("int local_ministep_num_active_data(local_ministep)")
    _active_data_list = ResPrototype(
        "active_list_ref local_ministep_get_active_data_list(local_ministep, char*)"
    )
    _has_active_data = ResPrototype(
        "bool local_ministep_data_is_active(local_ministep, char*)"
    )
    _add_active_data = ResPrototype(
        "void local_ministep_activate_data(local_ministep, char*)"
    )

    def __init__(self, ministep_key):
        raise NotImplementedError("Class can not be instantiated directly!")

    def set_ensemble_config(self, config):
        self.ensemble_config = config

    def hasActiveData(self, key):
        assert isinstance(key, str)
        return self._has_active_data(key)

    def addActiveData(self, key):
        assert isinstance(key, str)
        if key in self.ensemble_config:
            # TODO: Why bother? Why not make it idempotent?
            if not self._has_active_data(key):
                self._add_active_data(key)
            else:
                raise KeyError('Tried to add existing data key "%s".' % key)
        else:
            raise KeyError('Tried to add data key "%s" not in ensemble.' % key)

    def getActiveList(self, key):
        """@rtype: ActiveList"""
        if self._has_active_data(key):
            return self._active_data_list(key)
        else:
            raise KeyError('Local key "%s" not recognized.' % key)

    def numActiveData(self):
        return self._data_size()

    def addNode(self, node):
        assert isinstance(node, LocalObsdataNode)
        self._add_node(node)

    def attachObsset(self, obs_set):
        assert isinstance(obs_set, LocalObsdata)
        self._attach_obsdata(obs_set)

    def row_scaling(self, key) -> RowScaling:
        if not self._has_active_data(key):
            raise KeyError(f"Unknown key: {key}")

        return _lib.local.ministep.get_or_create_row_scaling(self, key)

    def getLocalObsData(self):
        """@rtype: LocalObsdata"""
        return self._get_local_obs_data()

    def name(self):
        return self._name()

    def getName(self):
        """@rtype: str"""
        return self.name()

    def get_obs_active_list(self) -> Dict[str, List[bool]]:
        return _lib.local.ministep.get_obs_active_list(self)

    def free(self):
        self._free()

    def __repr__(self):
        return "LocalMinistep(name = %s, len = %d) at 0x%x" % (
            self.name(),
            len(self),
            self._address(),
        )
