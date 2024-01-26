import os

import pytest


@pytest.fixture
def create_dummy_run_path(tmpdir):
    run_path = os.path.join(tmpdir, "out")
    os.mkdir(run_path)
    os.mkdir(os.path.join(run_path, "realization-0"))
    os.mkdir(os.path.join(run_path, "realization-0/iter-0"))
    os.mkdir(os.path.join(run_path, "realization-1"))
    os.mkdir(os.path.join(run_path, "realization-1/iter-0"))
    os.mkdir(os.path.join(run_path, "realization-1/iter-1"))
    yield os.chdir(tmpdir)
