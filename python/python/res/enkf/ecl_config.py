#  Copyright (C) 2012  Statoil ASA, Norway.
#
#  The file 'ecl_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.enkf import EnkfPrototype
from res.config import ConfigContent
from ecl.ecl import EclSum, EclGrid
from ecl.util import StringList, UIReturn
from res.sched import SchedFile

class EclConfig(BaseCClass):
    TYPE_NAME = "ecl_config"

    _alloc                  = EnkfPrototype("void* ecl_config_alloc(config_content)", bind = False)
    _free                   = EnkfPrototype("void  ecl_config_free( ecl_config )")
    _get_data_file          = EnkfPrototype("char* ecl_config_get_data_file(ecl_config)")
    _set_data_file          = EnkfPrototype("void  ecl_config_set_data_file(ecl_config , char*)")
    _validate_data_file     = EnkfPrototype("ui_return_obj ecl_config_validate_data_file(ecl_config , char*)")
    _get_gridfile           = EnkfPrototype("char* ecl_config_get_gridfile(ecl_config)")
    _set_gridfile           = EnkfPrototype("void  ecl_config_set_grid(ecl_config, char*)")
    _validate_gridfile      = EnkfPrototype("ui_return_obj ecl_config_validate_grid(ecl_config, char*)")
    _get_grid               = EnkfPrototype("ecl_grid_ref ecl_config_get_grid(ecl_config)")
    _get_schedule_file      = EnkfPrototype("char* ecl_config_get_schedule_file(ecl_config)")
    _set_schedule_file      = EnkfPrototype("void  ecl_config_set_schedule_file(ecl_config, char*, char*)")
    _validate_schedule_file = EnkfPrototype("ui_return_obj ecl_config_validate_schedule_file(ecl_config, char*)")
    _get_sched_file         = EnkfPrototype("sched_file_ref ecl_config_get_sched_file(ecl_config)")
    _get_init_section       = EnkfPrototype("char* ecl_config_get_init_section(ecl_config)")
    _set_init_section       = EnkfPrototype("void  ecl_config_set_init_section(ecl_config, char*)")
    _validate_init_section  = EnkfPrototype("ui_return_obj ecl_config_validate_init_section(ecl_config, char*)")
    _get_refcase_name       = EnkfPrototype("char* ecl_config_get_refcase_name(ecl_config)")
    _get_refcase            = EnkfPrototype("ecl_sum_ref ecl_config_get_refcase(ecl_config)")
    _load_refcase           = EnkfPrototype("void  ecl_config_load_refcase(ecl_config, char*)")
    _validate_refcase       = EnkfPrototype("ui_return_obj ecl_config_validate_refcase(ecl_config, char*)")
    _has_refcase            = EnkfPrototype("bool  ecl_config_has_refcase(ecl_config)")
    _get_depth_unit         = EnkfPrototype("char* ecl_config_get_depth_unit(ecl_config)")
    _get_pressure_unit      = EnkfPrototype("char* ecl_config_get_pressure_unit(ecl_config)")
    _get_start_date         = EnkfPrototype("time_t ecl_config_get_start_date(ecl_config)")
    _active                 = EnkfPrototype("bool ecl_config_active(ecl_config)")

    def __init__(self):
        c_ptr = self._alloc(None)
        if c_ptr:
            super(EclConfig, self).__init__(c_ptr)
        else:
            raise RuntimeError('Internal error: Failed constructing EclConfig!')

    def free(self):
        self._free()

    def getDataFile(self):
        return self._get_data_file()

    def setDataFile(self , datafile):
        self._set_data_file( datafile)

    def validateDataFile( self , datafile ):
        """ @rtype: UIReturn """
        return self._validate_data_file(  datafile )

    #-----------------------------------------------------------------

    def get_gridfile(self):
        """ @rtype: str """
        return self._get_gridfile()

    def set_gridfile(self, gridfile):
        self._set_gridfile(gridfile)

    def validateGridFile(self , gridfile):
        return self._validate_gridfile(gridfile)

    def getGrid(self):
        return self._get_grid()

    #-----------------------------------------------------------------

    def getScheduleFile(self):
        return self._get_schedule_file()

    def setScheduleFile(self, schedule_file, target_file = None):
        self._set_schedule_file(schedule_file, target_file)

    def validateScheduleFile(self , schedule_file):
        return self._validate_schedule_file(schedule_file)

    def get_sched_file(self):
        return self._get_sched_file()

    #-----------------------------------------------------------------

    def getInitSection(self):
        return self._get_init_section()

    def setInitSection(self, init_section):
        self._set_init_section(init_section)

    def validateInitSection(self, init_section):
        return self._validate_init_section(init_section)

    #-----------------------------------------------------------------

    def getRefcaseName(self):
        return self._get_refcase_name()

    def loadRefcase(self, refcase):
        self._load_refcase(refcase)

    def getRefcase(self):
        """ @rtype: EclSum """
        refcase = self._get_refcase()
        if not refcase is None:
            refcase.setParent(self)

        return refcase


    def validateRefcase(self, refcase):
        return self._validate_refcase(refcase)

    def hasRefcase(self):
        """ @rtype: bool """
        return self._has_refcase()

    #-----------------------------------------------------------------

    def getDepthUnit(self):
        return self._get_depth_unit()

    def getPressureUnit(self):
        return self._get_pressure_unit()

    #-----------------------------------------------------------------

    def getStartDate(self):
        return self._get_start_date()


    def active(self):
        """
        Has ECLIPSE been configured?"
        """
        return self._active( )
