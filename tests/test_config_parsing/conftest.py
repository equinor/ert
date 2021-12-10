import pytest


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield
