from pathlib import Path

import pytest

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow
from tests.utils import relpath, tmpdir

CONFIG_DIR = relpath("test_data", "mocked_test_case")
CONFIG_FILE = "config_workflow.yml"


@pytest.mark.integration_test
@tmpdir(CONFIG_DIR)
def test_workflow_run():
    config = EverestConfig.load_file(CONFIG_FILE)
    workflow = _EverestWorkflow(config)

    workflow.start_optimization()

    for name in ("pre_simulation", "post_simulation"):
        path = Path.cwd() / f"{name}.txt"
        assert path.exists()
        with path.open("r", encoding="utf-8") as file_obj:
            runpath = file_obj.readline()
        assert Path(runpath.strip()).exists()
