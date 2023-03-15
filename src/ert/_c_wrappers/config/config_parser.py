import os.path
from dataclasses import dataclass
from typing import List, Optional, Union

from cwrap import BaseCClass
from ecl.util.util import StringHash

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.config.config_content import ConfigContent
from ert._c_wrappers.config.unrecognized_enum import UnrecognizedEnum


@dataclass()
class Location:
    filename: str
    start_pos: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    end_pos: Optional[int] = None


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: str,
        config_file: Optional[str] = None,
        location: Optional[Union[str, Location]] = None,
    ) -> None:
        if config_file:
            self.location = Location(config_file)
        elif location is not None:
            if isinstance(location, Location):
                self.location = location
            else:
                self.location = Location(location)
        else:
            self.location = Location(filename="")

        self.errors = errors

        super().__init__(
            (
                f"Parsing config file `{self.config_file}` "
                f"resulted in the errors: {self.errors}"
            )
            if self.config_file
            else f"{self.errors}"
        )

    def replace(self, old_text: str, new_text: str):
        return ConfigValidationError(
            errors=self.errors.replace(old_text, new_text),
            config_file=self.config_file,
            location=self.location,
        )

    @property
    def config_file(self):
        return self.location.filename

    @config_file.setter
    def config_file(self, config_file):
        self.location.filename = config_file

    def get_error_messages(self):
        return [self.errors]


class CombinedConfigError(ConfigValidationError):
    def __init__(
        self,
        errors: Optional[
            List[Union[ConfigValidationError, "CombinedConfigError"]]
        ] = None,
    ):
        self.errors = []

        for err in errors or []:
            self.add_error(err)

    def __str__(self):
        return ", ".join(str(x) for x in self.errors)

    def is_empty(self):
        return len(self.errors) == 0

    def add_error(self, error: Union[ConfigValidationError, "CombinedConfigError"]):
        if isinstance(error, CombinedConfigError):
            self.errors.append(*error.errors)
        else:
            self.errors.append(error)

    def get_error_messages(self):
        all_messages = []
        for e in self.errors:
            all_messages.append(*e.get_error_messages())

        return all_messages

    def find_matching_error(self, match: str) -> Optional[ConfigValidationError]:
        return next(x for x in self.errors if match in str(x))

    @property
    def config_file(self):
        return self.errors[0].location.filename

    @config_file.setter
    def config_file(self, config_file: str):
        for err in self.errors:
            err.config_file = config_file


class ConfigParser(BaseCClass):
    TYPE_NAME = "config_parser"

    _alloc = ResPrototype("void* config_alloc()", bind=False)
    _add = ResPrototype(
        "schema_item_ref config_add_schema_item(config_parser, char*, bool)"
    )
    _free = ResPrototype("void config_free(config_parser)")
    _parse = ResPrototype(
        "config_content_obj config_parse(config_parser, \
                                         char*, \
                                         char*, \
                                         char*, \
                                         char*, \
                                         hash, \
                                         config_unrecognized_enum, \
                                         bool)"
    )
    _size = ResPrototype("int config_get_schema_size(config_parser)")
    _get_schema_item = ResPrototype(
        "schema_item_ref config_get_schema_item(config_parser, char*)"
    )
    _has_schema_item = ResPrototype("bool config_has_schema_item(config_parser, char*)")
    _add_key_value = ResPrototype(
        "bool config_parser_add_key_values(config_parser, \
                                           config_content, \
                                           char*, \
                                           stringlist, \
                                           config_path_elm, \
                                           char*, \
                                           config_unrecognized_enum)"
    )
    _validate = ResPrototype("void config_validate(config_parser, config_content)")

    def __init__(self):
        c_ptr = self._alloc()
        super().__init__(c_ptr)

    def __contains__(self, keyword):
        return self._has_schema_item(keyword)

    def __len__(self):
        return self._size()

    def __repr__(self):
        return self._create_repr(f"size={len(self)}")

    def add(self, keyword, required=False, value_type=None):
        item = self._add(keyword, required).setParent(self)
        if value_type:
            item.iset_type(0, value_type)
        return item

    def __getitem__(self, keyword):
        if keyword in self:
            item = self._get_schema_item(keyword)
            item.setParent(self)
            return item
        else:
            raise KeyError(f"Config parser does not have item:{keyword}")

    @staticmethod
    def check_non_utf_chars(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read()
        except UnicodeDecodeError as e:
            error_words = str(e).split(" ")
            hex_str = error_words[error_words.index("byte") + 1]
            try:
                unknown_char = chr(int(hex_str, 16))
            except ValueError:
                unknown_char = f"hex:{hex_str}"
            raise ConfigValidationError(
                f"Unsupported non UTF-8 character {unknown_char!r} "
                f"found in file: {file_path!r}",
                config_file=file_path,
            )

    def parse(
        self,
        config_file,
        comment_string="--",
        include_kw="INCLUDE",
        define_kw="DEFINE",
        pre_defined_kw_map=None,
        unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_WARN,
        validate=True,
    ) -> ConfigContent:
        assert isinstance(unrecognized, UnrecognizedEnum)

        hash_value = StringHash()
        if pre_defined_kw_map is not None:
            for key in pre_defined_kw_map:
                hash_value[key] = pre_defined_kw_map[key]

        if not os.path.exists(config_file):
            raise IOError(f"File: {config_file} does not exists")
        if not os.access(config_file, os.R_OK):
            raise IOError(f"We do not have read permissions for file: {config_file}")
        self.check_non_utf_chars(config_file)
        config_content = self._parse(
            config_file,
            comment_string,
            include_kw,
            define_kw,
            hash_value,
            unrecognized,
            validate,
        )
        config_content.setParser(self)

        if validate and not config_content.isValid():
            raise CombinedConfigError(
                errors=[
                    ConfigValidationError(errors=x, config_file=config_file)
                    for x in config_content.getErrors()
                ],
            )

        return config_content

    def free(self):
        self._free()

    def validate(self, config_content):
        self._validate(config_content)

    def add_key_value(
        self,
        config_content,
        key,
        value,
        path_elm=None,
        config_filename=None,
        unrecognized_action=UnrecognizedEnum.CONFIG_UNRECOGNIZED_WARN,
    ):
        return self._add_key_value(
            config_content, key, value, path_elm, config_filename, unrecognized_action
        )
