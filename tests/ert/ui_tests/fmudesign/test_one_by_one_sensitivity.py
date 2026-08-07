from importlib.resources import files

from ert.config.fmudesign import excel_to_dict
from tests.ert.ui_tests.fmudesign.design_input.fmudesign_ex_onebyone import (
    FmudesignOneByOne,
)

EXAMPLES_DIR = files("ert.config.fmudesign.examples")


def test_that_one_by_one_byte_stream_gives_same_dict_result_as_file():
    xlsx_byte_stream = FmudesignOneByOne().excel_byte_stream()

    stream_result = excel_to_dict(xlsx_byte_stream)

    one_by_one_example_filename = str(EXAMPLES_DIR / "fmudesign_ex_onebyone.xlsx")

    xlsx_result = excel_to_dict(one_by_one_example_filename)

    assert stream_result.keys() == xlsx_result.keys()
    assert all(
        stream_result[key] == xlsx_result[key]
        for key in xlsx_result
        # The inputfile will differ as one is a byte steam while the other a string
        if key != "input_file"
    )
