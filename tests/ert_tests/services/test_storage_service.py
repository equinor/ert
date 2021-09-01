import pytest
import requests
from ert_shared.services import Storage


@pytest.mark.requires_ert_storage
def test_integration(tmp_path, monkeypatch):
    """Actually start the server, wait for it to be online and do a health check"""
    monkeypatch.chdir(tmp_path)

    with Storage.start_server() as server:
        resp = requests.get(
            f"{server.fetch_url()}/healthcheck", auth=server.fetch_auth()
        )
        assert "ALL OK!" in resp.json()

        with Storage.session() as session:
            session.get("/healthcheck")

    assert not (tmp_path / "storage_server.json").exists()
