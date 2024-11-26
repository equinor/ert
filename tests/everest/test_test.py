import os

import pytest

from everest.config import EverestConfig
from tests.everest.generate_everest_configs import (
    generate_advanced_everest_config_file,
    generate_advanced_scipy_everest_config_file,
    generate_auto_scaled_controls_everest_config_file,
    generate_cvar_everest_config_file,
    generate_minimal_everest_config_file,
    generate_minimal_slow_everest_config_file,
    generate_multiobj_everest_config_file,
    generate_one_batch_everest_config_file,
    generate_remove_run_path_everest_config_file,
    generate_stddev_everest_config_file,
)


@pytest.mark.integration_test
def test_generate_minimal_config_same_as_existing_yml():
    config: EverestConfig = generate_minimal_everest_config_file()
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_minimal.yml")
    )
    assert config == stored_config


@pytest.mark.integration_test
def test_generate_advanced_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_advanced.yml")
    )
    config: EverestConfig = generate_advanced_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_advanced_scipy_config_same_as_existing_yml():
    path_to_config = os.path.abspath(
        os.path.join("test-data", "everest", "math_func", "config_advanced_scipy.yml")
    )
    stored_config = EverestConfig.load_file(path_to_config)
    stored_config.config_path = path_to_config

    config: EverestConfig = generate_advanced_scipy_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_auto_scaled_controls_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join(
            "test-data", "everest", "math_func", "config_auto_scaled_controls.yml"
        )
    )
    config: EverestConfig = generate_auto_scaled_controls_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_cvar_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_cvar.yml")
    )
    config: EverestConfig = generate_cvar_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_minimal_slow_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_minimal_slow.yml")
    )
    config: EverestConfig = generate_minimal_slow_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_multiobj_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_multiobj.yml")
    )
    config: EverestConfig = generate_multiobj_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_one_batch_config_same_as_existing_yml():
    path_to_config = os.path.abspath(
        os.path.join("test-data", "everest", "math_func", "config_one_batch.yml")
    )
    stored_config = EverestConfig.load_file(path_to_config)
    stored_config.config_path = path_to_config
    config: EverestConfig = generate_one_batch_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_remove_run_path_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_remove_run_path.yml")
    )
    config: EverestConfig = generate_remove_run_path_everest_config_file()

    assert config == stored_config


@pytest.mark.integration_test
def test_generate_stddev_config_same_as_existing_yml():
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_stddev.yml")
    )
    config: EverestConfig = generate_stddev_everest_config_file()

    assert config == stored_config
