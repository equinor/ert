import pytest

from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigWarning
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import get_optimal_result

CONFIG_FILE_ADVANCED = "config_advanced.yml"


@pytest.mark.slow
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_is_improvement_flag_handles_multiple_constraints(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file(CONFIG_FILE_ADVANCED)
    config_dict = {
        **config.model_dump(exclude_none=True),
        "output_constraints": [
            {"name": "x-0_coord", "lower_bound": 0.255, "upper_bound": 0.3},
            {"name": "x-1_coord", "lower_bound": 0.1, "upper_bound": 0.4},
        ],
    }
    config_dict["optimization"]["max_batch_num"] = 1
    with pytest.warns(ConfigWarning, match="The `controls.type` field is deprecated"):
        config = EverestConfig.model_validate(config_dict)

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        run_model = EverestRunModel.create(config, runtime_plugins=site_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    # The `is_improvement` flag should be False, no valid result:
    optimal_result = get_optimal_result(config.storage_dir)
    assert optimal_result is None
