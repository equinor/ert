import pytest

from ert._c_wrappers.enkf.export import ArgLoader


def test_arg_loader(tmp_path):

    with pytest.raises(IOError):
        arg = ArgLoader.load("arg1X")

    arg_file = tmp_path / "WI_1.txt"

    arg_file.write_text("1 2 3 4")

    with pytest.raises(ValueError):
        arg = ArgLoader.load(
            arg_file, column_names=["Col1", "Col2", "Col3", "COl5", "Col6"]
        )

    arg = ArgLoader.load(arg_file, column_names=["utm_x", "utm_y", "md", "tvd"])
    assert arg["utm_x"][0] == 1
