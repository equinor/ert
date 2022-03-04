from typing import Dict, List
from cwrap import BaseCClass

from res import _lib
from res import ResPrototype
from res.enkf.row_scaling import RowScaling
from res.enkf.local_obsdata import LocalObsdata


class LocalMinistep(BaseCClass):
    TYPE_NAME = "local_ministep"

    _free = ResPrototype("void local_ministep_free(local_ministep)")
    _name = ResPrototype("char* local_ministep_get_name(local_ministep)")
    _data_size = ResPrototype("int local_ministep_num_active_data(local_ministep)")
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
            return _lib.local.ministep.get_active_data_list(self, key)
        else:
            raise KeyError('Local key "%s" not recognized.' % key)

    def numActiveData(self):
        return self._data_size()

    def addNode(self, node):
        _lib.local.ministep.add_obsdata_node(self, node)

    def attachObsset(self, obs_set):
        assert isinstance(obs_set, LocalObsdata)
        _lib.local.ministep.attach_obsdata(self, obs_set)

    def row_scaling(self, key) -> RowScaling:
        if not self._has_active_data(key):
            raise KeyError(f"Unknown key: {key}")

        return _lib.local.ministep.get_or_create_row_scaling(self, key)

    def getLocalObsData(self):
        """@rtype: LocalObsdata"""
        return _lib.local.ministep.get_obsdata(self)

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
