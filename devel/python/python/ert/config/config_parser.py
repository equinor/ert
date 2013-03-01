#  Copyright (C) 2013  Statoil ASA, Norway. 
#   
#  The file 'config_parser.py' is part of ERT - Ensemble based Reservoir Tool. 
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

import libconfig
from   ert.cwrap.cwrap       import *
from   ert.cwrap.cclass      import CClass
import config_enums


class SchemaItem(CClass):
    
    def __init__(self , keyword , required = False):
        c_ptr = cfunc.schema_alloc( keyword , required )
        self.init_cref( c_ptr ,  cfunc.schema_free)

        
    @classmethod
    def wrap(cls , c_ptr , parent):
        obj = object.__new__( cls )
        obj.init_cref( c_ptr , parent )
        return obj


#-----------------------------------------------------------------


class ContentItem(CClass):
    # Not possible to create new python instances of this class
    
    @classmethod
    def wrap(cls , c_ptr , parent):
        obj = object.__new__( cls )
        obj.init_cref( c_ptr , parent )
        return obj
    


#-----------------------------------------------------------------




class ConfigParser(CClass):
    
    def __init__(self):
        c_ptr = cfunc.config_alloc()
        self.init_cobj(c_ptr , cfunc.config_free )

    
    def add(self , keyword , required = False):
        c_ptr = cfunc.add( self , keyword , required )
        schema_item = SchemaItem.wrap(c_ptr , self)
        return schema_item
    
    
    def parse( self , config_file , comment_string = "--" , include_kw = "INCLUDE" , define_kw = "DEFINE" , unrecognized = config_enums.unrecognized.CONFIG_UNRECOGNIZED_WARN , validate = True):
        return cfunc.parse( self , config_file , comment_string , include_kw , define_kw , unrecognized , validate )


    def __getitem__(self , keyword):
        if cfunc.has_content(self , keyword):
            c_ptr = cfunc.get_content(self , keyword )
            return ContentItem.wrap( c_ptr , self )
        else:
            raise KeyError("The config item:%s has not been set" % keyword)



cwrapper = CWrapper( libconfig.lib )
cwrapper.registerType( "config_parser" , ConfigParser )
cwrapper.registerType( "schema_item"   , SchemaItem )

cfunc = CWrapperNameSpace("config")

cfunc.add          = cwrapper.prototype("c_void_p config_add_schema_item( config_parser , char* , bool)")
cfunc.config_alloc = cwrapper.prototype("c_void_p config_alloc( )")
cfunc.config_free  = cwrapper.prototype("void config_free( config_parser )")
cfunc.parse        = cwrapper.prototype("bool config_parse( config_parser , char* , char* , char* , char* , int , bool )")
cfunc.has_content  = cwrapper.prototype("bool config_has_content_item( config_parser , char*) ")
cfunc.get_content  = cwrapper.prototype("c_void_p config_get_content_item( config_parser , char*) ")

cfunc.schema_free  = cwrapper.prototype("void config_schema_item_free( schema_item )")
cfunc.schema_alloc = cwrapper.prototype("c_void_p config_schema_item_alloc( char* , bool )")
