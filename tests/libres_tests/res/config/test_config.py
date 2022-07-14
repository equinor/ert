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

from ecl.util.test import TestAreaContext
from libres_utils import ResTest, tmpdir

from res import ResPrototype
from res.config import (
    ConfigParser,
    ContentItem,
    ContentNode,
    ContentTypeEnum,
    SchemaItem,
    UnrecognizedEnum,
)


class TestConfigPrototype(ResPrototype):
    def __init__(self, prototype, bind=False):
        super().__init__(prototype, bind=bind)


# Adding extra functions to the ConfigContent object for the ability
# to test low level C functions which are not exposed in Python.
_safe_iget = TestConfigPrototype(
    "char* config_content_safe_iget(config_content, char*, int, int)"
)
_iget = TestConfigPrototype(
    "char* config_content_iget(config_content, char*, int, int)"
)
_iget_as_int = TestConfigPrototype(
    "int config_content_iget_as_int(config_content, char*, int, int)"
)
_iget_as_bool = TestConfigPrototype(
    "bool config_content_iget_as_bool(config_content, char*, int, int)"
)
_iget_as_double = TestConfigPrototype(
    "double config_content_iget_as_double(config_content, char*, int, int)"
)
_get_occurences = TestConfigPrototype(
    "int config_content_get_occurences(config_content, char*)"
)


class ConfigTest(ResTest):
    def setUp(self):
        self.file_list = []

    def test_enums(self):
        source_file_path = "libres/lib/include/ert/config/config_schema_item.hpp"
        self.assertEnumIsFullyDefined(
            ContentTypeEnum, "config_item_types", source_file_path
        )
        self.assertEnumIsFullyDefined(
            UnrecognizedEnum, "config_schema_unrecognized_enum", source_file_path
        )

    def test_item_types(self):
        with TestAreaContext("config/types"):
            with open("config", "w") as f:
                f.write("TYPE_ITEM 10 3.14 TruE  String  file\n")

            conf = ConfigParser()
            self.assertEqual(0, len(conf))
            schema_item = conf.add("TYPE_ITEM", False)
            schema_item.iset_type(0, ContentTypeEnum.CONFIG_INT)
            schema_item.iset_type(1, ContentTypeEnum.CONFIG_FLOAT)
            schema_item.iset_type(2, ContentTypeEnum.CONFIG_BOOL)
            schema_item.iset_type(3, ContentTypeEnum.CONFIG_STRING)
            schema_item.iset_type(4, ContentTypeEnum.CONFIG_PATH)
            self.assertEqual(1, len(conf))
            self.assertNotIn("TYPE_XX", conf)
            self.assertIn("TYPE_ITEM", conf)

            content = conf.parse("config")
            type_item = content["TYPE_ITEM"][0]
            int_value = type_item[0]
            self.assertEqual(int_value, 10)
            self.assertEqual(type_item.igetString(0), "10")

            float_value = type_item[1]
            self.assertEqual(float_value, 3.14)
            self.assertEqual(type_item.igetString(1), "3.14")

            bool_value = type_item[2]
            self.assertEqual(bool_value, True)
            self.assertEqual(type_item.igetString(2), "TruE")

            string_value = type_item[3]
            self.assertEqual(string_value, "String")
            self.assertEqual(type_item.igetString(3), "String")

            path_value = type_item[4]
            self.assertEqual(path_value, os.path.abspath("file"))
            self.assertEqual(type_item.igetString(4), "file")

            # test __getitem__
            self.assertTrue(conf["TYPE_ITEM"])
            with self.assertRaises(KeyError):
                _ = conf["TYPE_XX"]

            self.assertIn("ConfigParser", repr(conf))
            self.assertIn("size=1", repr(conf))

    @tmpdir(None)
    def test_parse(self):
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
        self.assertIsInstance(schema_item, SchemaItem)
        content = conf.parse(
            "simple_config", unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_IGNORE
        )
        self.assertTrue(content.isValid())

        content_item = content["RSH_HOST"]
        self.assertIsInstance(content_item, ContentItem)
        self.assertEqual(len(content_item), 1)
        # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            content_item["BJARNE"]

        with self.assertRaises(IndexError):
            content_item[10]

        content_node = content_item[0]
        self.assertIsInstance(content_node, ContentNode)
        self.assertEqual(len(content_node), 2)
        self.assertEqual(content_node[1], "other-hostname:2")
        self.assertEqual(
            content_node.content(sep=","), "some-hostname:2,other-hostname:2"
        )
        self.assertEqual(content_node.content(), "some-hostname:2 other-hostname:2")

        content_item = content["FIELD"]
        self.assertEqual(len(content_item), 5)
        with self.assertRaises(IOError):
            conf.parse("DoesNotExits")

    def test_parse_invalid(self):
        conf = ConfigParser()
        conf.add("INT", value_type=ContentTypeEnum.CONFIG_INT)
        with TestAreaContext("config/parse2"):
            with open("config", "w") as fileH:
                fileH.write("INT xx\n")

            with self.assertRaises(ValueError):
                conf.parse("config")

            content = conf.parse("config", validate=False)
            self.assertFalse(content.isValid())
            self.assertEqual(len(content.getErrors()), 1)

    def test_parse_deprecated(self):
        conf = ConfigParser()
        item = conf.add("INT", value_type=ContentTypeEnum.CONFIG_INT)
        msg = "ITEM INT IS DEPRECATED"
        item.setDeprecated(msg)
        with TestAreaContext("config/parse2"):
            with open("config", "w") as fileH:
                fileH.write("INT 100\n")

            content = conf.parse("config")
            self.assertTrue(content.isValid())

            warnings = content.getWarnings()
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0], msg)

    def test_parse_dotdot_relative(self):
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
            self.assertEqual(script_path, node.getPath())

    def test_parser_content(self):
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
            self.assertTrue(content.isValid())
            self.assertTrue("KEY" in content)
            self.assertFalse("NOKEY" in content)
            self.assertEqual(cwd0, content.get_config_path())

            keys = content.keys()
            self.assertEqual(len(keys), 1)
            self.assertIn("KEY", keys)
            d = content.as_dict()
            self.assertIn("KEY", d)
            item_list = d["KEY"]
            self.assertEqual(len(item_list), 1)
            line = item_list[0]
            self.assertEqual(line[0], "VALUE1")
            self.assertEqual(line[1], "VALUE2")
            self.assertEqual(line[2], 100)
            self.assertEqual(line[3], True)
            self.assertEqual(line[4], 3.14)
            self.assertEqual(line[5], os.path.abspath("../path/file.txt"))

            self.assertFalse("NOT_IN_CONTENT" in content)
            item = content["NOT_IN_CONTENT"]
            self.assertEqual(len(item), 0)

            # pylint: disable=pointless-statement
            with self.assertRaises(KeyError):
                content["Nokey"]

            item = content["KEY"]
            self.assertEqual(len(item), 1)

            line = item[0]
            with self.assertRaises(TypeError):
                line.getPath(4)

            with self.assertRaises(TypeError):
                line.getPath()

            get = line[5]
            self.assertEqual(get, os.path.abspath("../path/file.txt"))
            abs_path = line.getPath(index=5)
            self.assertEqual(abs_path, os.path.join(cwd0, "path/file.txt"))

            with self.assertRaises(IndexError):
                item[10]

            node = item[0]
            self.assertEqual(len(node), 6)
            with self.assertRaises(IndexError):
                node[6]

            self.assertEqual(node[0], "VALUE1")
            self.assertEqual(node[1], "VALUE2")
            self.assertEqual(node[2], 100)
            self.assertEqual(node[3], True)
            self.assertEqual(node[4], 3.14)

            self.assertEqual(content.getValue("KEY", 0, 1), "VALUE2")
            self.assertEqual(_iget(content, "KEY", 0, 1), "VALUE2")

            self.assertEqual(content.getValue("KEY", 0, 2), 100)
            self.assertEqual(_iget_as_int(content, "KEY", 0, 2), 100)

            self.assertEqual(content.getValue("KEY", 0, 3), True)
            self.assertEqual(_iget_as_bool(content, "KEY", 0, 3), True)

            self.assertEqual(content.getValue("KEY", 0, 4), 3.14)
            self.assertEqual(_iget_as_double(content, "KEY", 0, 4), 3.14)

            self.assertIsNone(_safe_iget(content, "KEY2", 0, 0))

            self.assertEqual(_get_occurences(content, "KEY2"), 0)
            self.assertEqual(_get_occurences(content, "KEY"), 1)
            self.assertEqual(_get_occurences(content, "MISSING-KEY"), 0)

    def test_schema(self):
        schema_item = SchemaItem("TestItem")
        self.assertIsInstance(schema_item, SchemaItem)
        self.assertEqual(schema_item.iget_type(6), ContentTypeEnum.CONFIG_STRING)
        schema_item.iset_type(0, ContentTypeEnum.CONFIG_INT)
        self.assertEqual(schema_item.iget_type(0), ContentTypeEnum.CONFIG_INT)
        schema_item.set_argc_minmax(3, 6)

        del schema_item

    def test_add_unknown_keyowrds(self):
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

        self.assertIn("SETTINGS", content)
        item = content["SETTINGS"]
        self.assertEqual(len(item), 4)

        nodeA = item[0]
        self.assertEqual(nodeA[0], "A")
        self.assertEqual(nodeA[1], "100.1")
        self.assertEqual(len(nodeA), 2)

        nodeB = item[1]
        self.assertEqual(nodeB[0], "B")
        self.assertEqual(nodeB[1], "200")
        self.assertEqual(nodeB[3], "STRING2")
        self.assertEqual(len(nodeB), 4)

        self.assertEqual(len(content), 4)

    def test_valid_string_runtime_file(self):
        with TestAreaContext("assert_runtime_file"):
            with open("some_file", "w") as f:
                f.write("This i.")
            self.assertTrue(ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("no_file"))
            self.assertTrue(
                ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("some_file", True)
            )
            self.assertFalse(
                ContentTypeEnum.CONFIG_RUNTIME_FILE.valid_string("no_file", True)
            )

    def test_valid_string(self):
        self.assertTrue(ContentTypeEnum.CONFIG_FLOAT.valid_string("1.25"))
        self.assertTrue(ContentTypeEnum.CONFIG_RUNTIME_INT.valid_string("1.7"))
        self.assertFalse(
            ContentTypeEnum.CONFIG_RUNTIME_INT.valid_string("1.7", runtime=True)
        )
        self.assertTrue(
            ContentTypeEnum.CONFIG_FLOAT.valid_string("1.125", runtime=True)
        )
        self.assertEqual(ContentTypeEnum.CONFIG_FLOAT.convert_string("1.25"), 1.25)
        self.assertEqual(ContentTypeEnum.CONFIG_INT.convert_string("100"), 100)

        with self.assertRaises(ValueError):
            ContentTypeEnum.CONFIG_INT.convert_string("100x")

        with self.assertRaises(ValueError):
            ContentTypeEnum.CONFIG_FLOAT.convert_string("100X")

        with self.assertRaises(ValueError):
            ContentTypeEnum.CONFIG_BOOL.convert_string("a_random_string")

        self.assertTrue(ContentTypeEnum.CONFIG_BOOL.convert_string("TRUE"))
        self.assertTrue(ContentTypeEnum.CONFIG_BOOL.convert_string("True"))
        self.assertFalse(ContentTypeEnum.CONFIG_BOOL.convert_string("False"))
        self.assertFalse(ContentTypeEnum.CONFIG_BOOL.convert_string("F"))
