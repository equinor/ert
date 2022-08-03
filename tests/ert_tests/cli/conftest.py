import pytest

from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig


@pytest.fixture(autouse=True)
def no_cert_in_test(monkeypatch):
    # Do not generate certificates during test, parts of it can be time
    # consuming (e.g. 30 seconds)
    # Specifically generating the RSA key <_openssl.RSA_generate_key_ex>
    class MockESConfig(EvaluatorServerConfig):
        def __init__(self, *args, **kwargs):
            if "generate_cert" not in kwargs:
                kwargs["generate_cert"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("ert_shared.cli.main.EvaluatorServerConfig", MockESConfig)
