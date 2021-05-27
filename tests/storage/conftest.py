import pytest


@pytest.fixture(autouse=True)
def _enable_new_storage(monkeypatch):
    """
    All tests in this module assume --enable-new-storage is set
    """
    from ert_shared.feature_toggling import FeatureToggling

    feature = FeatureToggling._conf["new-storage"]
    monkeypatch.setattr(feature, "is_enabled", True)


@pytest.fixture
def _disable_server_monitor(monkeypatch):
    """
    The ServerMonitor singleton starts ert-storage as a subprocess. Mock it to not block.
    """
    from ert_shared.storage import extraction

    class MockServerMonitor:
        @classmethod
        def get_instance(cls):
            return cls()

        @staticmethod
        def fetch_url():
            return ""

        @staticmethod
        def fetch_auth():
            return ("", "")

    monkeypatch.setattr(extraction, "ServerMonitor", MockServerMonitor)


@pytest.fixture
def client(_disable_server_monitor, monkeypatch, ert_storage_client):
    import requests

    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "ON")
    # Fix requests library
    for func in "get", "post", "put", "delete":
        monkeypatch.setattr(requests, func, getattr(ert_storage_client, func))

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
