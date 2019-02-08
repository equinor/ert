#  Copyright (C) 2012  Equinor ASA, Norway.
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
from res import ResPrototype
from res.config import ConfigSettings



class PlotSettings(ConfigSettings):
    TYPE_NAME    = "plot_settings"

    _alloc       = ResPrototype("void* plot_settings_alloc(config_content)", bind=False)
    _init        = ResPrototype("void plot_settings_init(plot_settings)")

    def __init__(self, config_content = None):
        if config_content:
            c_ptr = self._alloc(config_content)

            if c_ptr is None:
                raise ValueError('Failed to construct RNGConfig instance')

            super(PlotSettings, self).__init__("PLOT_SETTING", c_ptr)
        else:
            super(PlotSettings, self).__init__("PLOT_SETTING")
            self._init( )

    def getPath(self):
        """ @rtype: str """
        return self["PATH"]

    def setPath(self, path):
        self["PATH"] = path
