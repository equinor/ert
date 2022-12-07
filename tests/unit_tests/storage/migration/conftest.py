import pytest


@pytest.fixture(scope="session")
def block_storage_path(source_root):
    path = source_root / "test-data/block_storage/snake_oil"
    if not path.is_dir():
        pytest.skip(
            "'test-data/block_storage' has not been checked out.\n"
            "Run: git submodule update --init --recursive"
        )
    return path.parent
