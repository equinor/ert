from pathlib import Path
from typing import Callable, Optional

import pytest

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow
from tests.everest.utils import relpath, skipif_no_everest_models

CONFIG_DIR = relpath("test_data", "mocked_test_case")
CONFIG_FILE = "config_workflow.yml"


@pytest.mark.integration_test
def test_workflow_run(copy_mocked_test_data_to_tmp):
    config = EverestConfig.load_file(CONFIG_FILE)
    workflow = _EverestWorkflow(config)

    workflow.start_optimization()

    for name in ("pre_simulation", "post_simulation"):
        path = Path.cwd() / f"{name}.txt"
        assert path.exists()
        with path.open("r", encoding="utf-8") as file_obj:
            runpath = file_obj.readline()
        assert Path(runpath.strip()).exists()


@pytest.mark.integration_test
@pytest.mark.everest_models_test
@skipif_no_everest_models
@pytest.mark.parametrize("config", ("array", "index"))
def test_state_modifier_workflow_run(
    config: str, copy_testdata_tmpdir: Callable[[Optional[str]], Path]
) -> None:
    cwd = copy_testdata_tmpdir("open_shut_state_modifier")
    _EverestWorkflow(
        config=EverestConfig.load_file(f"everest/model/{config}.yml")
    ).start_optimization()

    for path in Path.cwd().glob("**/simulation_0/RESULT.SCH"):
        assert path.read_bytes() == (cwd / "eclipse/model/EXPECTED.SCH").read_bytes()
