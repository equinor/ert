#  Copyright (C) 2012  Statoil ASA, Norway.
#
#  The file 'plot_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.enkf import EnkfPrototype

class PlotConfig(BaseCClass):
    TYPE_NAME = "plot_config"
    #cwrapper.registerType("plot_config_obj", PlotConfig.createPythonObject)
    #cwrapper.registerType("plot_config_ref", PlotConfig.createCReference)

    _free     = EnkfPrototype("void  plot_config_free( plot_config )")
    _get_path = EnkfPrototype("char* plot_config_get_path(plot_config)")
    _set_path = EnkfPrototype("void  plot_config_set_path(plot_config, char*)")

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getPath(self):
        """ @rtype: str """
        return self._get_path()

    def setPath(self, path):
        self._set_path(path)

    def free(self):
        self._free()
