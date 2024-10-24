from pathlib import Path
from typing import Callable, Optional

import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import relpath, skipif_no_everest_models

CONFIG_DIR = relpath("test_data", "mocked_test_case")
CONFIG_FILE = "config_workflow.yml"


@pytest.mark.integration_test
def test_workflow_run(copy_mocked_test_data_to_tmp, evaluator_server_config_generator):
    config = EverestConfig.load_file(CONFIG_FILE)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

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
    config: str,
    copy_testdata_tmpdir: Callable[[Optional[str]], Path],
    evaluator_server_config_generator,
) -> None:
    cwd = copy_testdata_tmpdir("open_shut_state_modifier")

    run_model = EverestRunModel.create(
        EverestConfig.load_file(f"everest/model/{config}.yml")
    )
    evaluator_server_config = evaluator_server_config_generator(run_model)
    run_model.run_experiment(evaluator_server_config)

    for path in Path.cwd().glob("**/simulation_0/RESULT.SCH"):
        assert path.read_bytes() == (cwd / "eclipse/model/EXPECTED.SCH").read_bytes()
