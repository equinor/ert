#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'test_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os

import pytest
from ecl.util.test import TestAreaContext

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.config import (
    ConfigParser,
    ContentItem,
    ContentNode,
    ContentTypeEnum,
    SchemaItem,
    UnrecognizedEnum,
)


class _TestConfigPrototype(ResPrototype):
    def __init__(self, prototype, bind=False):
        super().__init__(prototype, bind=bind)


# Adding extra functions to the ConfigContent object for the ability
# to test low level C functions which are not exposed in Python.
_safe_iget = _TestConfigPrototype(
    "char* config_content_safe_iget(config_content, char*, int, int)"
)
_iget = _TestConfigPrototype(
    "char* config_content_iget(config_content, char*, int, int)"
)
_iget_as_int = _TestConfigPrototype(
    "int config_content_iget_as_int(config_content, char*, int, int)"
)
_iget_as_bool = _TestConfigPrototype(
    "bool config_content_iget_as_bool(config_content, char*, int, int)"
)
_iget_as_double = _TestConfigPrototype(
    "double config_content_iget_as_double(config_content, char*, int, int)"
)
_get_occurences = _TestConfigPrototype(
    "int config_content_get_occurences(config_content, char*)"
)


def test_item_types(tmp_path):
    with open(tmp_path / "config", "w") as f:
        f.write("TYPE_ITEM 10 3.14 TruE  String  file\n")

    conf = ConfigParser()
    assert len(conf) == 0
    schema_item = conf.add("TYPE_ITEM", False)
    schema_item.iset_type(0, ContentTypeEnum.CONFIG_INT)
    schema_item.iset_type(1, ContentTypeEnum.CONFIG_FLOAT)
    schema_item.iset_type(2, ContentTypeEnum.CONFIG_BOOL)
    schema_item.iset_type(3, ContentTypeEnum.CONFIG_STRING)
    schema_item.iset_type(4, ContentTypeEnum.CONFIG_PATH)
    assert len(conf) == 1
    assert "TYPE_XX" not in conf
    assert "TYPE_ITEM" in conf

    content = conf.parse(str(tmp_path / "config"))
    type_item = content["TYPE_ITEM"][0]
    assert type_item[0] == 10
    assert type_item.igetString(0) == "10"

    assert type_item[1] == 3.14
    assert type_item.igetString(1) == "3.14"

    assert type_item[2] is True
    assert type_item.igetString(2) == "TruE"

    assert type_item[3] == "String"
    assert type_item.igetString(3) == "String"

    assert type_item[4] == str(tmp_path / "file")
    assert type_item.igetString(4) == "file"

    # test __getitem__
    assert conf["TYPE_ITEM"]
    with pytest.raises(KeyError):
        _ = conf["TYPE_XX"]

    assert "ConfigParser" in repr(conf)
    assert "size=1" in repr(conf)


@pytest.mark.usefixtures("use_tmpdir")
def test_parse():
    config_file = """
RSH_HOST some-hostname:2 other-hostname:2
FIELD    PRESSURE      DYNAMIC
FIELD    SWAT          DYNAMIC   MIN:0   MAX:1
FIELD    SGAS          DYNAMIC   MIN:0   MAX:1
FIELD    RS            DYNAMIC   MIN:0
FIELD    RV            DYNAMIC   MIN:0.0034"""
    with open("simple_config", "w") as fout:
        fout.write(config_file)
    conf = ConfigParser()
    conf.add("FIELD", False)
    schema_item = conf.add("RSH_HOST", False)
    assert isinstance(schema_item, SchemaItem)
    content = conf.parse(
        "simple_config", unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_IGNORE
    )
    assert content.isValid()

    content_item = content["RSH_HOST"]
    assert isinstance(content_item, ContentItem)
    assert len(content_item) == 1
    # pylint: disable=pointless-statement
    with pytest.raises(TypeError):
        content_item["BJARNE"]

    with pytest.raises(IndexError):
        content_item[10]

    content_node = content_item[0]
    assert isinstance(content_node, ContentNode)
    assert len(content_node) == 2
    assert content_node[1] == "other-hostname:2"
    assert content_node.content(sep=",") == "some-hostname:2,other-hostname:2"
    assert content_node.content() == "some-hostname:2 other-hostname:2"

    content_item = content["FIELD"]
    assert len(content_item) == 5
    with pytest.raises(IOError):
        conf.parse("DoesNotExits")


def test_parse_invalid():
    conf = ConfigParser()
    conf.add("INT", value_type=ContentTypeEnum.CONFIG_INT)
    with TestAreaContext("config/parse2"):
        with open("config", "w") as fileH:
            fileH.write("INT xx\n")

        with pytest.raises(ValueError):
            conf.parse("config")

        content = conf.parse("config", validate=False)
        assert not content.isValid()
        assert len(content.getErrors()) == 1


def test_parse_deprecated():
    conf = ConfigParser()
    item = conf.add("INT", value_type=ContentTypeEnum.CONFIG_INT)
    msg = "ITEM INT IS DEPRECATED"
    item.setDeprecated(msg)
    with TestAreaContext("config/parse2"):
        with open("config", "w") as fileH:
            fileH.write("INT 100\n")

        content = conf.parse("config")
        assert content.isValid()

        warnings = content.getWarnings()
        assert len(warnings) == 1
        assert warnings[0] == msg


def test_parse_dotdot_relative():
    conf = ConfigParser()
    schema_item = conf.add("EXECUTABLE", False)
    schema_item.iset_type(0, ContentTypeEnum.CONFIG_PATH)

    with TestAreaContext("config/parse_dotdot"):
        os.makedirs("cwd/jobs")
        os.makedirs("eclipse/bin")
        script_path = os.path.join(os.getcwd(), "eclipse/bin/script.sh")
        with open(script_path, "w") as f:
            f.write("This is a test script")

        with open("cwd/jobs/JOB", "w") as fileH:
            fileH.write("EXECUTABLE ../../eclipse/bin/script.sh\n")

        os.makedirs("cwd/ert")
        os.chdir("cwd/ert")
        content = conf.parse("../jobs/JOB")
        item = content["EXECUTABLE"]
        node = item[0]
        assert script_path == node.getPath()


def test_parser_content():
    conf = ConfigParser()
    conf.add("KEY2", False)
    schema_item = conf.add("KEY", False)
    schema_item.iset_type(2, ContentTypeEnum.CONFIG_INT)
    schema_item.iset_type(3, ContentTypeEnum.CONFIG_BOOL)
    schema_item.iset_type(4, ContentTypeEnum.CONFIG_FLOAT)
    schema_item.iset_type(5, ContentTypeEnum.CONFIG_PATH)
    schema_item = conf.add("NOT_IN_CONTENT", False)

    with TestAreaContext("config/parse2"):
        with open("config", "w") as fileH:
            fileH.write("KEY VALUE1 VALUE2 100  True  3.14  path/file.txt\n")

        cwd0 = os.getcwd()
        os.makedirs("tmp")
        os.chdir("tmp")
        content = conf.parse("../config")
        d = content.as_dict()
        assert content.isValid()
        assert "KEY" in content
        assert "NOKEY" not in content
        assert cwd0 == content.get_config_path()

        keys = content.keys()
        assert len(keys) == 1
        assert "KEY" in keys
        d = content.as_dict()
        assert "KEY" in d
        item_list = d["KEY"]
        assert len(item_list) == 1
        line = item_list[0]
        assert line[0] == "VALUE1"
        assert line[1] == "VALUE2"
        assert line[2] == 100
        assert line[3] is True
        assert line[4] == 3.14
        assert line[5] == os.path.abspath("../path/file.txt")

        assert "NOT_IN_CONTENT" not in content
        item = content["NOT_IN_CONTENT"]
        assert len(item) == 0

        # pylint: disable=pointless-statement
        with pytest.raises(KeyError):
            content["Nokey"]

        item = content["KEY"]
        assert len(item) == 1

        line = item[0]
        with pytest.raises(TypeError):
            line.getPath(4)

        with pytest.raises(TypeError):
            line.getPath()

        get = line[5]
        assert get == os.path.abspath("../path/file.txt")
        abs_path = line.getPath(index=5)
        assert abs_path == os.path.join(cwd0, "path/file.txt")

        with pytest.raises(IndexError):
            item[10]

        node = item[0]
        assert len(node) == 6
        with pytest.raises(IndexError):
            node[6]

        assert node[0] == "VALUE1"
        assert node[1] == "VALUE2"
        assert node[2] == 100
        assert node[3] is True
        assert node[4] == 3.14

        assert content.getValue("KEY", 0, 1) == "VALUE2"
        assert _iget(content, "KEY", 0, 1) == "VALUE2"

        assert content.getValue("KEY", 0, 2) == 100
        assert _iget_as_int(content, "KEY", 0, 2) == 100

        assert content.getValue("KEY", 0, 3) is True
        assert _iget_as_bool(content, "KEY", 0, 3) is True

        assert content.getValue("KEY", 0, 4) == 3.14
        assert _iget_as_double(content, "KEY", 0, 4) == 3.14

        assert _safe_iget(content, "KEY2", 0, 0) is None

        assert _get_occurences(content, "KEY2") == 0
        assert _get_occurences(content, "KEY") == 1
        assert _get_occurences(content, "MISSING-KEY") == 0


def test_schema():
    schema_item = SchemaItem("TestItem")
    assert isinstance(schema_item, SchemaItem)
    assert schema_item.iget_type(6) == ContentTypeEnum.CONFIG_STRING
    schema_item.iset_type(0, ContentTypeEnum.CONFIG_INT)
    assert schema_item.iget_type(0) == ContentTypeEnum.CONFIG_INT
    schema_item.set_argc_minmax(3, 6)

    del schema_item


def test_add_unknown_keyowrds():
    parser = ConfigParser()
    with TestAreaContext("config/parse4"):
        with open("config", "w") as fileH:
            fileH.write("SETTINGS A 100.1\n")
            fileH.write("SETTINGS B 200  STRING1 STRING2\n")
            fileH.write("SETTINGS C 300\n")
            fileH.write("SETTINGS D False\n")

        content = parser.parse(
            "config", unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_ADD
        )

    assert "SETTINGS" in content
    item = content["SETTINGS"]
    assert len(item) == 4

    nodeA = item[0]
    assert nodeA[0] == "A"
    assert nodeA[1] == "100.1"
    assert len(nodeA) == 2

    nodeB = item[1]
    assert nodeB[0] == "B"
    assert nodeB[1] == "200"
    assert nodeB[3] == "STRING2"
    assert len(nodeB) == 4

    assert len(content) == 4


def test_valid_string_runtime_file():
    with TestAreaContext("assert_runtime_file"):
        with open("some_file", "w") as f:
            f.write("This i.")
        assert ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("no_file")
        assert ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("some_file", True)
        assert not ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("no_file", True)


def test_valid_string():
    assert ContentTypeEnum.CONFIG_FLOAT.valid_string("1.25")
    assert ContentTypeEnum.CONFIG_RUNTIME_INT.valid_string("1.7")
    assert not ContentTypeEnum.CONFIG_RUNTIME_INT.valid_string("1.7", runtime=True)
    assert ContentTypeEnum.CONFIG_FLOAT.valid_string("1.125", runtime=True)
    assert ContentTypeEnum.CONFIG_FLOAT.convert_string("1.25") == 1.25
    assert ContentTypeEnum.CONFIG_INT.convert_string("100") == 100

    with pytest.raises(ValueError):
        ContentTypeEnum.CONFIG_INT.convert_string("100x")

    with pytest.raises(ValueError):
        ContentTypeEnum.CONFIG_FLOAT.convert_string("100X")

    with pytest.raises(ValueError):
        ContentTypeEnum.CONFIG_BOOL.convert_string("a_random_string")

    assert ContentTypeEnum.CONFIG_BOOL.convert_string("TRUE")
    assert ContentTypeEnum.CONFIG_BOOL.convert_string("True")
    assert not ContentTypeEnum.CONFIG_BOOL.convert_string("False")
    assert not ContentTypeEnum.CONFIG_BOOL.convert_string("F")
