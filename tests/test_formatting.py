import sys
import os

import pytest
from tests.conftest import source_root
from tests.excluded_files_black import get_files_excluded_from_black


def get_py_files(path):
    fnames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                fnames.append(os.path.join(root, file))
    return fnames


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires Python3")
def test_code_style(source_root):
    from click.testing import CliRunner
    import black

    excluded_files = get_files_excluded_from_black(source_root)
    py_files_ert_shared = get_py_files(os.path.join(source_root, "ert_shared"))
    py_files_ert_gui = get_py_files(os.path.join(source_root, "ert_gui"))
    py_files_tests = get_py_files(os.path.join(source_root, "tests"))
    py_files_test_data = get_py_files(os.path.join(source_root, "test-data"))
    files_to_test = set(
        py_files_ert_gui + py_files_ert_shared + py_files_test_data + py_files_tests
    ) - set(excluded_files)

    runner = CliRunner()

    resp = runner.invoke(black.main, ["--check"] + list(files_to_test))
    assert (
        resp.exit_code == 0
    ), "Black would still reformat one or more files:\n{}".format(resp.output)
