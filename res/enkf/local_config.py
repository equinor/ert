#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'local_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res import ResPrototype
from res import _lib
from res.analysis import AnalysisModule
from res.enkf.local_ministep import LocalMinistep
from res.enkf.local_updatestep import LocalUpdateStep


class LocalConfig(BaseCClass):
    """The LocalConfig class is created as a reference to an existing underlying C
    structure by the method EnkFMain.local_config(). When the pointer to the C
    local_config_type object has been properly wrapped we 'decorate' the Python
    object with references to the ensemble_config , observations and grid.

    This implies that the Python object LocalConfig is richer than the
    underlying C object local_config_type; the extra attributes are only used
    for validation.

    """

    TYPE_NAME = "local_config"

    _free = ResPrototype("void   local_config_free(local_config)")
    _clear = ResPrototype("void   local_config_clear(local_config)")
    _clear_active = ResPrototype("void   local_config_clear_active(local_config)")
    _create_ministep = ResPrototype(
        "local_ministep_ref local_config_alloc_ministep"
        "(local_config, char*, analysis_module)"
    )
    _attach_ministep = ResPrototype(
        "void   local_updatestep_add_ministep(local_updatestep, local_ministep)",
        bind=False,
    )
    _has_obsdata = ResPrototype("bool   local_config_has_obsdata(local_config, char*)")

    _get_updatestep = ResPrototype(
        "local_updatestep_ref local_config_get_updatestep(local_config)"
    )
    _get_ministep = ResPrototype(
        "local_ministep_ref   local_config_get_ministep(local_config, char*)"
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def initAttributes(self, ensemble_config, grid):
        self.ensemble_config = ensemble_config
        self.grid = grid

    def __getEnsembleConfig(self):
        return self.ensemble_config

    def getGrid(self):
        # The grid can be None
        return self.grid

    def free(self):
        self._free()

    def clear(self):
        self._clear()

    def clear_active(self):
        self._clear_active()

    def createMinistep(self, mini_step_key, analysis_module=None):
        """@rtype: Ministep"""
        assert isinstance(mini_step_key, str)
        if analysis_module:
            assert isinstance(analysis_module, AnalysisModule)
        ministep = self._create_ministep(mini_step_key, analysis_module)
        if ministep is None:
            raise KeyError("Ministep:  {} already exists".format(mini_step_key))
        ministep.set_ensemble_config(self.__getEnsembleConfig())
        return ministep

    def createObsdata(self, obsdata_key):
        """@rtype: Obsdata"""
        assert isinstance(obsdata_key, str)
        if self._has_obsdata(obsdata_key):
            raise ValueError("Tried to add existing observation key:%s " % obsdata_key)

        return _lib.local.local_config.create_obsdata(self, obsdata_key)

    def copyObsdata(self, src_key, target_key):
        """@rtype: Obsdata"""
        assert isinstance(src_key, str)
        assert isinstance(target_key, str)
        if not self._has_obsdata(src_key):
            raise KeyError(f"The observation set {src_key} does not exist")

        obsdata = _lib.local.local_config.get_obsdata_copy(self, src_key, target_key)
        return obsdata

    def getUpdatestep(self):
        """@rtype: UpdateStep"""
        return self._get_updatestep()

    def getMinistep(self, mini_step_key):
        """@rtype: Ministep"""
        assert isinstance(mini_step_key, str)
        return self._get_ministep(mini_step_key)

    def getObsdata(self, obsdata_key):
        """@rtype: Obsdata"""
        assert isinstance(obsdata_key, str)
        return _lib.local.local_config.get_obsdata_ref(self, obsdata_key)

    def attachMinistep(self, update_step, mini_step):
        assert isinstance(mini_step, LocalMinistep)
        assert isinstance(update_step, LocalUpdateStep)
        self._attach_ministep(update_step, mini_step)

    def __repr__(self):
        return self._create_repr()
