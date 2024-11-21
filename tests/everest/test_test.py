import os

import pytest

from everest.config import EverestConfig
from tests.everest.generate_everest_configs import generate_minimal_everest_config


@pytest.mark.integration_test
def test_generate_minimal_config_same_as_existing_yml():
    config: EverestConfig = generate_minimal_everest_config()
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_minimal.yml")
    )
    assert config == stored_config
