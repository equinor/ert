import os
import shutil
import pytest

from res.enkf import ResConfig


@pytest.fixture()
def setup_case(tmpdir, source_root):
    def copy_case(path, config_file):
        shutil.copytree(os.path.join(source_root, "test-data", path), "test_data")
        os.chdir("test_data")
        return ResConfig(config_file)

    with tmpdir.as_cwd():
        yield copy_case
