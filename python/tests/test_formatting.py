import sys
import os

import pytest
from tests.conftest import source_root
from tests.excluded_files import get_excluded_files


def get_py_files(path):
    fnames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                fnames.append(os.path.join(root, file))
    return fnames


def test_code_style():
    from click.testing import CliRunner
    import black

    root = os.path.join(source_root(), "python")

    excluded_files = get_excluded_files(root)
    all_py_files = get_py_files(root)
    files_to_test = set(all_py_files) - set(excluded_files)

    runner = CliRunner()

    resp = runner.invoke(black.main, ["--check"] + list(files_to_test))
    assert (
        resp.exit_code == 0
    ), "Black would still reformat one or more files:\n{}".format(resp.output)
