from ecl.util.util import StringList

from ert._c_wrappers.config.config_content import ConfigContent
from ert._c_wrappers.config.config_parser import ConfigParser
from ert._c_wrappers.config.content_type_enum import ContentTypeEnum


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
