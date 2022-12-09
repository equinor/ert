from typing import TYPE_CHECKING, Iterator, List

from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.config.config_path_elm import ConfigPathElm
from ert._c_wrappers.config.content_type_enum import ContentTypeEnum
from ert._c_wrappers.config.schema_item import SchemaItem
from ert._c_wrappers.util.substitution_list import SubstitutionList

if TYPE_CHECKING:
    from ert._c_wrappers.config.config_parser import ConfigParser


class ContentNode(BaseCClass):
    TYPE_NAME = "content_node"

    _iget = ResPrototype("char* config_content_node_iget( content_node , int)")
    _size = ResPrototype("int config_content_node_get_size( content_node )")
    _get_full_string = ResPrototype(
        "char* config_content_node_get_full_string( content_node , char* )"
    )
    _iget_type = ResPrototype(
        "config_content_type_enum config_content_node_iget_type( content_node , int)"
    )
    _iget_as_abspath = ResPrototype(
        "char* config_content_node_iget_as_abspath( content_node , int)"
    )
    _iget_as_string = ResPrototype(
        "char* config_content_node_iget( content_node , int)"
    )
    _iget_as_int = ResPrototype(
        "int config_content_node_iget_as_int( content_node , int)"
    )
    _iget_as_double = ResPrototype(
        "double config_content_node_iget_as_double( content_node , int)"
    )
    _iget_as_path = ResPrototype(
        "char* config_content_node_iget_as_path( content_node , int)"
    )
    _iget_as_executable = ResPrototype(
        "char* config_content_node_iget_as_executable( content_node , int)"
    )
    _iget_as_bool = ResPrototype(
        "bool config_content_node_iget_as_bool( content_node , int)"
    )
    _iget_as_isodate = ResPrototype(
        "time_t config_content_node_iget_as_isodate( content_node , int)"
    )

    _free = ResPrototype("void config_content_node_free( content_node )")

    typed_get = {
        ContentTypeEnum.CONFIG_STRING: _iget_as_string,
        ContentTypeEnum.CONFIG_INT: _iget_as_int,
        ContentTypeEnum.CONFIG_FLOAT: _iget_as_double,
        ContentTypeEnum.CONFIG_PATH: _iget_as_path,
        ContentTypeEnum.CONFIG_EXISTING_PATH: _iget_as_path,
        ContentTypeEnum.CONFIG_BOOL: _iget_as_bool,
        ContentTypeEnum.CONFIG_ISODATE: _iget_as_isodate,
        ContentTypeEnum.CONFIG_EXECUTABLE: _iget_as_executable,
    }

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __len__(self):
        return self._size()

    def __iter__(self):
        return iter([self[i] for i in range(len(self))])

    def __assertIndex(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if not 0 <= index < len(self):
                raise IndexError
            return index
        else:
            raise TypeError(f"Invalid argument type: {index}")

    def __getitem__(self, index):
        index = self.__assertIndex(index)

        content_type = self._iget_type(index)
        typed_get = self.typed_get[content_type]
        return typed_get(self, index)

    def getPath(self, index=0):
        index = self.__assertIndex(index)
        content_type = self._iget_type(index)
        if content_type in [
            ContentTypeEnum.CONFIG_EXISTING_PATH,
            ContentTypeEnum.CONFIG_PATH,
        ]:
            return self._iget_as_abspath(index)
        else:
            raise TypeError("The getPath() method can only be called on PATH items")

    def content(self, sep=" "):
        return self._get_full_string(sep)

    def igetString(self, index):
        index = self.__assertIndex(index)
        return self._iget(index)

    def free(self):
        self._free()


class ContentItem(BaseCClass):
    TYPE_NAME = "content_item"

    _alloc = ResPrototype(
        "void* config_content_item_alloc( schema_item , void* )", bind=False
    )
    _size = ResPrototype("int config_content_item_get_size( content_item )")
    _iget_content_node = ResPrototype(
        "content_node_ref config_content_item_iget_node( content_item , int)"
    )
    _free = ResPrototype("void config_content_item_free( content_item )")

    def __init__(self, schema_item):
        path_elm = None
        c_ptr = self._alloc(schema_item, path_elm)
        super().__init__(c_ptr)

    def __len__(self):
        return self._size()

    def __iter__(self) -> Iterator[ContentNode]:
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index: int) -> ContentNode:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if 0 <= index < len(self):
                return self._iget_content_node(index).setParent(self)
            else:
                raise IndexError(
                    f"Expected 0 <= index < {len(self)}, was 0 <= {index} < {len(self)}"
                )
        else:
            raise TypeError("[] operator must have integer index")

    def last(self):
        return self[-1]

    def getValue(self, item_index=-1, node_index=0):
        node = self[item_index]
        return node[node_index]

    def free(self):
        self._free()


class ConfigContent(BaseCClass):
    TYPE_NAME = "config_content"

    _alloc = ResPrototype("void* config_content_alloc(char*)", bind=False)
    _free = ResPrototype("void config_content_free( config_content )")
    _is_valid = ResPrototype("bool config_content_is_valid( config_content )")
    _has_key = ResPrototype("bool config_content_has_item( config_content , char*)")
    _get_item = ResPrototype(
        "content_item_ref config_content_get_item( config_content , char*)"
    )
    _get_warnings = ResPrototype(
        "stringlist_ref config_content_get_warnings( config_content )"
    )
    _get_config_path = ResPrototype(
        "char* config_content_get_config_path( config_content )"
    )
    _create_path_elm = ResPrototype(
        "config_path_elm_ref config_content_add_path_elm(config_content, char*)"
    )
    _get_const_define_list = ResPrototype(
        "subst_list_ref config_content_get_const_define_list(config_content)"
    )
    _add_define = ResPrototype(
        "void config_content_add_define(config_content, char*, char*)"
    )
    _size = ResPrototype("int config_content_get_size(config_content)")
    _keys = ResPrototype("stringlist_obj config_content_alloc_keys(config_content)")

    def __init__(self, filename):
        c_ptr = self._alloc(filename)

        self._parser = None  # To be set later

        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Failed to construct ConfigContent "
                f"instance from config file {filename}."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n\t"
            + ",\n\t".join([f"{k}: {self.getValue(k)}" for k in self.keys()])
            + ")"
        )

    def __len__(self):
        return self._size()

    def __contains__(self, key: str) -> bool:
        return self._has_key(key)

    def setParser(self, parser: "ConfigParser"):
        self._parser = parser

    def __getitem__(self, key: str) -> ContentItem:
        if key in self:
            item = self._get_item(key)
            item.setParent(self)
            return item
        else:
            if self._parser is not None and key in self._parser:
                schema_item = SchemaItem(key)
                return ContentItem(schema_item)
            else:
                raise KeyError("No such key: {key}")

    def hasKey(self, key):
        return key in self

    def getValue(self, key, item_index=-1, node_index=0):
        item = self[key]
        return item.getValue(item_index, node_index)

    def isValid(self):
        return self._is_valid()

    def free(self):
        self._free()

    def getErrors(self) -> List[str]:
        return _clib.config_content.get_errors(self)

    def getWarnings(self) -> List[str]:
        return list(self._get_warnings())

    def get_config_path(self) -> str:
        return self._get_config_path()

    def create_path_elm(self, path) -> ConfigPathElm:
        return self._create_path_elm(path)

    def get_const_define_list(self) -> SubstitutionList:
        return self._get_const_define_list()

    def add_define(self, key: str, value: str):
        self._add_define(key, value)

    def keys(self) -> List[str]:
        return list(self._keys())
