import pytest


@pytest.fixture
def client(monkeypatch, ert_storage_client):
    from ert.shared.storage import extraction

    class MockStorageService:
        @staticmethod
        def session():
            return ert_storage_client

    monkeypatch.setattr(extraction, "StorageService", MockStorageService)
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "ON")

    # Store a list of experiment IDs that exist in the database, in case the
    # database isn't clean prior to running tests.
    pre_experiments = {
        exp["id"] for exp in ert_storage_client.get("/experiments").json()
    }

    def fetch_experiment():
        experiments = [
            exp["id"]
            for exp in ert_storage_client.get("/experiments").json()
            if exp["id"] not in pre_experiments
        ]
        assert len(experiments) == 1
        return experiments[0]

    ert_storage_client.fetch_experiment = fetch_experiment
    return ert_storage_client
