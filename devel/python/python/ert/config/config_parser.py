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

import os.path

from ert.config import UnrecognizedEnum, CONFIG_LIB, ContentTypeEnum , ConfigContent
from ert.cwrap import BaseCClass, CWrapper


class SchemaItem(BaseCClass):
    def __init__(self, keyword, required=False):
        c_ptr = SchemaItem.cNamespace().alloc(keyword, required)
        super(SchemaItem, self).__init__(c_ptr)


    def iget_type( self, index):
        """ @rtype: ContentTypeEnum """
        return SchemaItem.cNamespace().iget_type(self, index)

    def iset_type( self, index, schema_type ):
        """
        @type schema_type: ContentTypeEnum
        """
        assert isinstance(schema_type, ContentTypeEnum)
        SchemaItem.cNamespace().iset_type(self, index, schema_type)

    def set_argc_minmax(self, minimum, maximum):
        SchemaItem.cNamespace().set_argc_minmax(self, minimum, maximum)

    def free(self):
        SchemaItem.cNamespace().free(self)



<<<<<<< HEAD


class ConfigParser(BaseCClass):
=======
class ContentItem(BaseCClass):
    # Not possible to create new python instances of this class
    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")


    def __len__(self):
        return ContentItem.cNamespace().size(self)


    def __getitem__(self, index):
        if isinstance(index, int):
            if (index >= 0) and (index < self.__len__()):
                return ContentItem.cNamespace().iget_content_node(self, index).setParent(self)
            else:
                raise IndexError
        else:
            raise TypeError("[] operator must have integer index")


class ContentNode(BaseCClass):
    typed_get = {}
    
>>>>>>> ERT-747: Added type conversion to config get
    def __init__(self):
        c_ptr = ConfigParser.cNamespace().alloc()
        super(ConfigParser, self).__init__(c_ptr)

    
    def __contains__(self , keyword):
        return ConfigParser.cNamespace().has_schema_item( self , keyword )

<<<<<<< HEAD

    def add(self, keyword, required=False):
        return ConfigParser.cNamespace().add(self, keyword, required).setParent( self )
=======
    def __assertIndex__(self , index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)
                
            if not 0 <= index < len(self):
                raise IndexError
        else:
            raise TypeError("Invalid argument type: %s" % index)

                
    def __getitem__(self, index):
        self.__assertIndex__(index)
        
        content_type = ContentNode.cNamespace().iget_type(self, index)
        typed_get = self.typed_get[content_type]
        return typed_get( self , index )

        
    def content(self, sep=" "):
        return ContentNode.cNamespace().get_full_string(self, sep)
>>>>>>> ERT-747: Added type conversion to config get


    def igetString(self , index):
        self.__assertIndex__(index)
        return ContentNode.cNamespace().iget(self , index )


    def asList(self):
        return [x for x in self]


    def getSchemaItem(self , keyword):
        if keyword in self:
            item = ConfigParser.cNamespace().get_schema_item(self, keyword)
            item.setParent( self ) 
        else:
            raise KeyError("Config parser does not have item:%s" % keyword)


    def parse(self, config_file, comment_string="--", include_kw="INCLUDE", define_kw="DEFINE",
               unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_WARN, validate=True):

        assert isinstance(unrecognized, UnrecognizedEnum)
        if os.path.exists(config_file):
            config_content = ConfigParser.cNamespace().parse(self, config_file, comment_string, include_kw, define_kw, unrecognized, validate)
            if config_content.isValid():
                return config_content
            else:
                raise Exception("Parsing:%s failed" % config_file)
        else:
            raise IOError("File: %s does not exists")


    def free(self):
        ConfigParser.cNamespace().free(self)


cwrapper = CWrapper(CONFIG_LIB)
cwrapper.registerObjectType("config_parser", ConfigParser)
cwrapper.registerObjectType("schema_item", SchemaItem)


ConfigParser.cNamespace().alloc = cwrapper.prototype("c_void_p config_alloc( )")
ConfigParser.cNamespace().add = cwrapper.prototype("schema_item_ref config_add_schema_item( config_parser , char* , bool)")
ConfigParser.cNamespace().free = cwrapper.prototype("void config_free( config_parser )")
ConfigParser.cNamespace().parse = cwrapper.prototype("config_content_obj config_parse( config_parser , char* , char* , char* , char* , config_unrecognized_enum , bool )")
ConfigParser.cNamespace().get_schema_item = cwrapper.prototype("schema_item_ref config_get_schema_item( config_parser , char*)")
ConfigParser.cNamespace().has_schema_item = cwrapper.prototype("bool config_has_schema_item( config_parser , char*)")

SchemaItem.cNamespace().alloc = cwrapper.prototype("c_void_p config_schema_item_alloc( char* , bool )")
SchemaItem.cNamespace().free = cwrapper.prototype("void config_schema_item_free( schema_item )")
SchemaItem.cNamespace().iget_type = cwrapper.prototype("config_content_type_enum config_schema_item_iget_type( schema_item, int)")
SchemaItem.cNamespace().iset_type = cwrapper.prototype("void config_schema_item_iset_type( schema_item , int , config_content_type_enum)")
SchemaItem.cNamespace().set_argc_minmax = cwrapper.prototype("void config_schema_item_set_argc_minmax( schema_item , int , int)")


