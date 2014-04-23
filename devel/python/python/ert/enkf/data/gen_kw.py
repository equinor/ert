#  Copyright (C) 2012  Statoil ASA, Norway. 
#   
#  The file 'gen_kw.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.cwrap import BaseCClass, CWrapper, CFILE
from ert.enkf import ENKF_LIB


class GenKw(BaseCClass):
    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def exportParameters(self, file_name):
        """ @type: str """
        py_fileH = open(file_name , "w")
        cfile  = CFILE( py_fileH )
        GenKw.cNamespace().export_parameters(self, cfile)
        py_fileH.close()

    def exportTemplate(self, file_name):
        """ @type: str """
        GenKw.cNamespace().export_template(self, file_name)

    def free(self):
        GenKw.cNamespace().free(self)


    def getNode(self, gen_kw_config):
        """ @rtype: GenKw """
        return GenKw.cNamespace().alloc(gen_kw_config)

    ##################################################################

cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("gen_kw", GenKw)
cwrapper.registerType("gen_kw_obj", GenKw.createPythonObject)
cwrapper.registerType("gen_kw_ref", GenKw.createCReference)

GenKw.cNamespace().free = cwrapper.prototype("void gen_kw_free(gen_kw_config)")
GenKw.cNamespace().alloc = cwrapper.prototype("gen_kw_obj gen_kw_alloc(gen_kw_config)")
GenKw.cNamespace().export_parameters = cwrapper.prototype("void gen_kw_write_export_file(gen_kw , FILE)")
GenKw.cNamespace().export_template = cwrapper.prototype("void gen_kw_ecl_write_template(gen_kw , char* )")
