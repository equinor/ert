import os

import pytest

from everest.config import EverestConfig


@pytest.mark.integration_test
def test_generate_minimal_config_same_as_existing_yml(minimal_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_minimal.yml")
    )
    assert minimal_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_advanced_config_same_as_existing_yml(advanced_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_advanced.yml")
    )

    assert advanced_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_advanced_scipy_config_same_as_existing_yml(
    advanced_scipy_everest_config,
):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_advanced_scipy.yml")
    )

    assert advanced_scipy_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_auto_scaled_controls_config_same_as_existing_yml(
    auto_scaled_controls_everest_config,
):
    stored_config = EverestConfig.load_file(
        os.path.join(
            "test-data", "everest", "math_func", "config_auto_scaled_controls.yml"
        )
    )

    assert auto_scaled_controls_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_cvar_config_same_as_existing_yml(cvar_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_cvar.yml")
    )

    assert cvar_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_minimal_slow_config_same_as_existing_yml(minimal_slow_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_minimal_slow.yml")
    )

    assert minimal_slow_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_multiobj_config_same_as_existing_yml(multiobj_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_multiobj.yml")
    )

    assert multiobj_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_one_batch_config_same_as_existing_yml(one_batch_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_one_batch.yml")
    )

    assert one_batch_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_remove_run_path_config_same_as_existing_yml(
    remove_run_path_everest_config,
):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_remove_run_path.yml")
    )

    assert remove_run_path_everest_config == stored_config


@pytest.mark.integration_test
def test_generate_stddev_config_same_as_existing_yml(stddev_everest_config):
    stored_config = EverestConfig.load_file(
        os.path.join("test-data", "everest", "math_func", "config_stddev.yml")
    )

    assert stddev_everest_config == stored_config
