import copy
import stat
from pathlib import Path
from shutil import which
from textwrap import dedent

import pytest

from ert.config import ConfigWarning
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.plugins import ErtPluginContext
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


@pytest.mark.integration_test
@pytest.mark.parametrize("test_deprecated", [True, False])
def test_workflow_will_run_during_experiment(
    min_config, test_deprecated, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    workflow_job_script_content = dedent("""#!/usr/bin/env python

import argparse
import sys


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-o", "--out", type=str, required=True)
    arg_parser.add_argument("-m", "--message", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    msg = options.message or "test"
    with open(options.out, "w", encoding="utf-8") as f:
        f.write(f"{msg}\\n")


if __name__ == "__main__":
    main(sys.argv[1:])

    """)

    wf_path = Path("test_wf.py")
    wf_path.write_text(workflow_job_script_content, encoding="utf-8")
    wf_path.chmod(wf_path.stat().st_mode | stat.S_IEXEC)

    min_config["install_workflow_jobs"] = [
        {"name": "test_wf", "executable": "test_wf.py"}
    ]

    min_config["workflows"] = {
        "pre_simulation": [
            f"test_wf -o {tmp_path}/pre_simulation.txt -m <RUNPATH_FILE>"
        ],
        "post_simulation": [
            f"test_wf -o {tmp_path}/post_simulation.txt -m <RUNPATH_FILE>"
        ],
    }

    min_config["forward_model"] = ["sleep 1"]
    min_config["install_jobs"] = [{"name": "sleep", "executable": which("sleep")}]

    min_config["config_path"] = Path.cwd() / "config.yml"
    Path("config.yml").touch()

    if test_deprecated:
        deprecated_config_dict = copy.deepcopy(min_config)
        deprecated_config_dict["install_workflow_jobs"][0].pop("executable")

        Path("TEST_WF").write_text("EXECUTABLE test_wf.py", encoding="utf-8")
        deprecated_config_dict["install_workflow_jobs"][0]["source"] = "TEST_WF"

        with pytest.warns(
            ConfigWarning, match="`install_workflow_jobs: source` is deprecated"
        ):
            config = EverestConfig.model_validate(deprecated_config_dict)
    else:
        config = EverestConfig.model_validate(min_config)

    with ErtPluginContext() as runtime_plugins:
        run_model = EverestRunModel.create(config, runtime_plugins=runtime_plugins)

    evaluator_server_config = EvaluatorServerConfig()
    run_model.run_experiment(evaluator_server_config)

    for name in ("pre_simulation", "post_simulation"):
        output_file_path = Path.cwd() / f"{name}.txt"
        assert output_file_path.exists()

        with output_file_path.open("r", encoding="utf-8") as file_obj:
            runpath_content = file_obj.readline().strip()

        assert Path(runpath_content).exists()
