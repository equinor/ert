from ecl.util.util import StringList

from ert._c_wrappers.config.config_content import ConfigContent
from ert._c_wrappers.config.config_parser import ConfigParser
from ert.parsing import SchemaItemType


def test_get_executable_list(tmpdir):
    parser = ConfigParser()
    parser.add("MYKEYWORD", value_type=SchemaItemType.EXECUTABLE)

    content = ConfigContent(None)
    path_elm = content.create_path_elm(str(tmpdir))
    content.setParser(parser)

    values = ["MY_EXECUTABLE", "MY_EXECUTABLE2", "PATH/MY_EXECUTABLE"]

    for value in values:
        parser.add_key_value(
            content,
            "MYKEYWORD",
            StringList(["MYKEYWORD", value]),
            path_elm=path_elm,
        )

    my_keyword_sets = list(content["MYKEYWORD"])
    assert len(my_keyword_sets) == 3

    errors = list(content.getErrors())
    for value, error in zip(values, errors):
        expected_error = f"Executable:{value} does not exist"
        assert expected_error == error
