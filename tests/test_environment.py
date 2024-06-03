import os

import everest
import pytest
from everest.config import EverestConfig
from everest.simulator.everest2res import everest2res

from tests.utils import relpath, tmpdir

root = os.path.join("..", "examples", "math_func")
config_file = "config_minimal.yml"


@pytest.mark.integration_test
@tmpdir(relpath(root))
def test_seed():
    random_seed = 42
    config = EverestConfig.load_file(config_file)
    config.environment.random_seed = random_seed

    ever_workflow = everest.suite._EverestWorkflow(config)

    assert random_seed == ever_workflow.config.environment.random_seed

    # Res
    ert_config = everest2res(config)
    assert random_seed == ert_config["RANDOM_SEED"]


@pytest.mark.integration_test
@tmpdir(relpath(root))
def test_loglevel():
    config = EverestConfig.load_file(config_file)
    config.environment.log_level = "info"
    ever_workflow = everest.suite._EverestWorkflow(config)
    config = ever_workflow.config
    assert len(EverestConfig.lint_config_dict(config.to_dict())) == 0
