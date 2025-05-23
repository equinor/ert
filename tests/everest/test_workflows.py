from collections.abc import Callable
from pathlib import Path

import pytest

from ert.config import ConfigWarning
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from tests.everest.utils import skipif_no_everest_models

CONFIG_FILE = "config_workflow.yml"


@pytest.mark.integration_test
@pytest.mark.parametrize("test_deprecated", [True, False])
def test_workflow_will_run_during_experiment(
    copy_mocked_test_data_to_tmp, test_deprecated
):
    config = EverestConfig.load_file(CONFIG_FILE)
    if test_deprecated:
        config_dict = config.to_dict()
        del config_dict["install_workflow_jobs"][0]["executable"]
        config_dict["install_workflow_jobs"][0]["source"] = "jobs/TEST_WF"
        with pytest.warns(
            ConfigWarning, match="`install_workflow_jobs: source` is deprecated"
        ):
            config = EverestConfig.model_validate(config_dict)

    run_model = EverestRunModel.create(config)
    evaluator_server_config = EvaluatorServerConfig()
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
    copy_testdata_tmpdir: Callable[[str | None], Path],
) -> None:
    cwd = copy_testdata_tmpdir("open_shut_state_modifier")

    run_model = EverestRunModel.create(
        EverestConfig.load_file(f"everest/model/{config}.yml")
    )
    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)
    paths = list(Path.cwd().glob("**/simulation_0/RESULT.SCH"))
    assert paths
    for path in paths:
        assert path.read_bytes() == (cwd / "eclipse/model/EXPECTED.SCH").read_bytes()
