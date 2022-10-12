from pathlib import Path

from ecl.util.util import StringList

from ert._c_wrappers.config.config_content import ConfigContent
from ert._c_wrappers.config.config_parser import ConfigParser
from ert._c_wrappers.config.content_type_enum import ContentTypeEnum
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._clib.config_keywords import init_user_config_parser


def test_get_executable_list(tmpdir):
    parser = ConfigParser()
    parser.add("MYKEYWORD", value_type=ContentTypeEnum.CONFIG_EXECUTABLE)

    content = ConfigContent(None)
    path_elm = content.create_path_elm(str(tmpdir))
    content.setParser(parser)

    parser.add_key_value(
        content,
        "MYKEYWORD",
        StringList(["MYKEYWORD", "MY_EXECUTABLE"]),
        path_elm=path_elm,
    )
    parser.add_key_value(
        content,
        "MYKEYWORD",
        StringList(["MYKEYWORD", "MY_EXECUTABLE2"]),
        path_elm=path_elm,
    )

    my_keyword_sets = list(content["MYKEYWORD"])
    assert len(my_keyword_sets) == 2

    errors = list(content.getErrors())
    assert "MY_EXECUTABLE" in errors[0]
    assert " does not exist" in errors[0]
    assert "MY_EXECUTABLE2" in errors[1]
    assert " does not exist" in errors[1]


def test_config_content_as_dict(tmpdir):
    with tmpdir.as_cwd():
        conf = ConfigParser()
        existing_file_1 = "test_1.t"
        existing_file_2 = "test_2.t"
        Path(existing_file_2).write_text("something")
        Path(existing_file_1).write_text("not important")
        init_user_config_parser(conf)

        schema_item = conf.add("MULTIPLE_KEY_VALUE", False)
        schema_item.iset_type(0, ContentTypeEnum.CONFIG_INT)

        schema_item = conf.add("KEY", False)
        schema_item.iset_type(2, ContentTypeEnum.CONFIG_INT)

        with open("config", "w") as fileH:
            fileH.write(f"{ConfigKeys.NUM_REALIZATIONS} 42\n")
            fileH.write(f"DATA_FILE {existing_file_2} \n")
            fileH.write(f"DATA_FILE {existing_file_1} \n")
            fileH.write(f"REFCASE {existing_file_1} \n")

            fileH.write("MULTIPLE_KEY_VALUE 6\n")
            fileH.write("MULTIPLE_KEY_VALUE 24\n")
            fileH.write("MULTIPLE_KEY_VALUE 12\n")
            fileH.write("QUEUE_OPTION SLURM MAX_RUNNING 50\n")
            fileH.write("KEY VALUE1 VALUE1 100\n")
            fileH.write("KEY VALUE2 VALUE2 200\n")
        content = conf.parse("config")
        content_as_dict = content.as_dict()
        assert content_as_dict == {
            "KEY": [["VALUE1", "VALUE1", 100], ["VALUE2", "VALUE2", 200]],
            ConfigKeys.NUM_REALIZATIONS: 42,
            ConfigKeys.QUEUE_OPTION: [["SLURM", "MAX_RUNNING", "50"]],
            "MULTIPLE_KEY_VALUE": [6, 24, 12],
            ConfigKeys.DATA_FILE: [
                str(Path.cwd() / existing_file_2),
                str(Path.cwd() / existing_file_1),
            ],
            ConfigKeys.REFCASE: str(Path.cwd() / existing_file_1),
        }
