import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict

CONFIG_FILE = "config_minimal.yml"


@pytest.mark.integration_test
def test_seed(copy_math_func_test_data_to_tmp):
    random_seed = 42
    config = EverestConfig.load_file(CONFIG_FILE)
    config.environment.random_seed = random_seed

    run_model = EverestRunModel.create(config)
    assert random_seed == run_model.everest_config.environment.random_seed

    # Res
    ert_config = _everest_to_ert_config_dict(config)
    assert random_seed == ert_config["RANDOM_SEED"]


@pytest.mark.integration_test
def test_loglevel(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    config.environment.log_level = "info"
    run_model = EverestRunModel.create(config)
    config = run_model.everest_config
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0
