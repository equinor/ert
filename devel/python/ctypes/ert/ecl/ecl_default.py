#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_default.py' is part of ERT - Ensemble based Reservoir Tool. 
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
"""
Module containing default values for ECLIPSE.

This module contains site specific default values for various
variables related to the eclipse environment. Observe that this module
itself does not set any default values, instead it will try to import
a module ecl_local; and then subsequently read all values from this
module. Observe that the local module is not provided in the ert
source distribution, you must write this yourself.

It is perfectly permissible to not provide a ecl_local module, but if
you at a later stage try to use a property from the EclConfig class
you will get an exception.
""" 

#cmd                  = None
#version              = None
#lsf_resource_request = None

#cmd                  = "/project/res/etc/ERT/Scripts/run_eclipse.py"
#version              = "2009.2"   
#lsf_resource_request = "select[cs && x86_64Linux] rusage[ecl100v2000=1:duration=5]"


class EclConfig:
    def __init__(self):
        pass

    def safe_get( self , attr ):
        if hasattr( self , "__%s" % attr):
            value = getattr( self , "__%s" % attr)
            return value
        else:
            raise NotImplementedError("The default attribute:%s has not been set - you must update/provide a ecl_local module." % attr)
        
    @property
    def cmd( self ):
        return self.safe_get( "cmd" )

    @property
    def version( self ):
        return self.safe_get( "version" )

    @property
    def lsf_resource_request( self ):
        return self.safe_get( "lsf_resource_request" )


try:
    import ecl_local

    if hasattr( ecl_local , "cmd"):
        EclConfig.__cmd = ecl_local.cmd

    if hasattr( ecl_local , "version"):
        EclConfig.__version = ecl_local.version

    if hasattr( ecl_local , "lsf_resource_request"):
        EclConfig.__lsf_resource_request = ecl_local.lsf_resource_request

except ImportError:
    pass


config = EclConfig()

