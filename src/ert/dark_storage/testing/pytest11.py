import pytest
from typing import Generator, TYPE_CHECKING
from ert_storage.testing import testclient_factory


if TYPE_CHECKING:
    from ert_storage.testing.testclient import _TestClient


@pytest.fixture
def ert_storage_client() -> Generator["_TestClient", None, None]:
    with testclient_factory() as testclient:
        yield testclient
