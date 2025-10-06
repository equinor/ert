import json
import os
import os.path
import stat
import textwrap
from pathlib import Path

import pytest

from _ert.forward_model_runner.reporting.message import Checksum, Exited, Start
from _ert.forward_model_runner.runner import ForwardModelRunner
from ert.config import ErtConfig, ForwardModelStep
from ert.config.ert_config import (
    create_forward_model_json,
    forward_model_step_from_config_contents,
)


def create_jobs_json(fm_step_list):
    return {"jobList": fm_step_list}


@pytest.fixture(autouse=True)
def set_up_environ():
    if "ERT_RUN_ID" in os.environ:
        del os.environ["ERT_RUN_ID"]

    yield

    keys = (
        "KEY_ONE",
        "KEY_TWO",
        "KEY_THREE",
        "KEY_FOUR",
        "PATH104",
        "ERT_RUN_ID",
    )

    for key in keys:
        if key in os.environ:
            del os.environ[key]


@pytest.mark.usefixtures("use_tmpdir")
def test_missing_joblist_json():
    with pytest.raises(KeyError):
        ForwardModelRunner({})


@pytest.mark.usefixtures("use_tmpdir")
def test_run_output_rename():
    fm_step = {
        "name": "TEST_FMSTEP",
        "executable": "mkdir",
        "stdout": "out",
        "stderr": "err",
    }
    fm_step_list = [fm_step, fm_step, fm_step, fm_step, fm_step]

    fmr = ForwardModelRunner(create_jobs_json(fm_step_list))

    for status in enumerate(fmr.run([])):
        if isinstance(status, Start):
            assert status.job is not None
            assert status.job.std_err == f"err.{status.job.index}"
            assert status.job.std_out == f"out.{status.job.index}"


@pytest.mark.usefixtures("use_tmpdir")
def test_run_multiple_ok():
    fm_step_list = []
    dir_list = ["1", "2", "3", "4", "5"]
    for fm_step_index in dir_list:
        fm_step = {
            "name": "MKDIR",
            "executable": "mkdir",
            "stdout": f"mkdir_out.{fm_step_index}",
            "stderr": f"mkdir_err.{fm_step_index}",
            "argList": ["-p", "-v", fm_step_index],
        }
        fm_step_list.append(fm_step)

    fmr = ForwardModelRunner(create_jobs_json(fm_step_list))

    statuses = [s for s in list(fmr.run([])) if isinstance(s, Exited)]

    assert len(statuses) == 5
    for status in statuses:
        assert status.exit_code == 0

    for dir_number in dir_list:
        assert os.path.isdir(dir_number)
        assert os.path.isfile(f"mkdir_out.{dir_number}")
        assert os.path.isfile(f"mkdir_err.{dir_number}")
        assert os.path.getsize(f"mkdir_err.{dir_number}") == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_when_forward_model_contains_multiple_steps_just_one_checksum_status_is_given():
    fm_step_list = []
    file_list = ["1", "2", "3", "4", "5"]
    manifest = {}
    for fm_step_index in file_list:
        manifest[f"file_{fm_step_index}"] = fm_step_index
        fm_step = {
            "name": "TOUCH",
            "executable": "touch",
            "stdout": f"touch_out.{fm_step_index}",
            "stderr": f"touch_err.{fm_step_index}",
            "argList": [fm_step_index],
        }
        fm_step_list.append(fm_step)
    Path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fmr = ForwardModelRunner(create_jobs_json(fm_step_list))

    statuses = [s for s in list(fmr.run([])) if isinstance(s, Checksum)]
    assert len(statuses) == 1
    assert len(statuses[0].data) == 5


@pytest.mark.usefixtures("use_tmpdir")
def test_when_manifest_file_is_not_created_by_fm_runner_checksum_contains_error():
    fm_step_list = []
    file_name = "test"
    manifest = {"file_1": f"{file_name}"}

    fm_step_list.append(
        {
            "name": "TOUCH",
            "executable": "touch",
            "stdout": "touch_out.test",
            "stderr": "touch_err.test",
            "argList": ["not_test"],
        }
    )
    Path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fmr = ForwardModelRunner(create_jobs_json(fm_step_list))

    checksum_msg = [s for s in list(fmr.run([])) if isinstance(s, Checksum)]
    assert len(checksum_msg) == 1
    info = checksum_msg[0].data["file_1"]
    assert "md5sum" not in info
    assert "error" in info
    assert (
        f"Expected file {os.getcwd()}/{file_name} not created by forward model!"
        in info["error"]
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_multiple_fail_only_runs_one():
    fm_step_list = []
    for index in range(1, 6):
        fm_step = {
            "name": "exit",
            "executable": "/bin/sh",
            "stdout": "exit_out",
            "stderr": "exit_err",
            # produces something on stderr, and exits with
            "argList": [
                "-c",
                f'echo "failed with {index}" 1>&2 ; exit {index}',
            ],
        }
        fm_step_list.append(fm_step)

    fmr = ForwardModelRunner(create_jobs_json(fm_step_list))

    statuses = [s for s in list(fmr.run([])) if isinstance(s, Exited)]

    assert len(statuses) == 1
    for i, status in enumerate(statuses):
        assert status.exit_code == i + 1


@pytest.mark.usefixtures("use_tmpdir")
def test_env_var_available_inside_step_context():
    Path("run_me.py").write_text(
        textwrap.dedent(
            """\
                #!/usr/bin/env python
                import os
                assert os.environ["TEST_ENV"] == "123"
                """
        ),
        encoding="utf-8",
    )
    os.chmod("run_me.py", stat.S_IEXEC + stat.S_IREAD)

    step = forward_model_step_from_config_contents(
        """
        EXECUTABLE run_me.py
        ENV TEST_ENV 123
        """,
        name=None,
        config_file="RUN_ENV",
    )
    with open("jobs.json", mode="w", encoding="utf-8") as fptr:
        ert_config = ErtConfig(forward_model_steps=[step])
        json.dump(
            create_forward_model_json(
                context=ert_config.substitutions,
                forward_model_steps=ert_config.forward_model_steps,
                env_vars=ert_config.env_vars,
                user_config_file=ert_config.user_config_file,
                run_id="run_id",
            ),
            fptr,
        )

    with open("jobs.json", encoding="utf-8") as f:
        jobs_json = json.load(f)

    # Check ENV variable not available outside of step context
    assert "TEST_ENV" not in os.environ

    for msg in list(ForwardModelRunner(jobs_json).run([])):
        if isinstance(msg, Exited):
            assert msg.exit_code == 0

    # Check ENV variable not available outside of step context
    assert "TEST_ENV" not in os.environ


@pytest.mark.usefixtures("use_tmpdir")
def test_default_env_variables_available_inside_fm_step_context():
    Path("run_me.py").write_text(
        textwrap.dedent(
            """\
                #!/usr/bin/env python
                import os
                assert os.environ["_ERT_ITERATION_NUMBER"] == "0"
                assert os.environ["_ERT_REALIZATION_NUMBER"] == "0"
                assert os.environ["_ERT_RUNPATH"] == "./"
                """
        ),
        encoding="utf-8",
    )
    os.chmod("run_me.py", stat.S_IEXEC + stat.S_IREAD)

    step = forward_model_step_from_config_contents(
        "EXECUTABLE run_me.py", name=None, config_file="RUN_ENV"
    )
    with open("jobs.json", mode="w", encoding="utf-8") as fptr:
        ert_config = ErtConfig(
            forward_model_steps=[step],
            substitutions={"<RUNPATH>": "./"},
        )
        json.dump(
            create_forward_model_json(
                context=ert_config.substitutions,
                forward_model_steps=ert_config.forward_model_steps,
                env_vars=ert_config.env_vars,
                user_config_file=ert_config.user_config_file,
                run_id="run_id",
            ),
            fptr,
        )

    with open("jobs.json", encoding="utf-8") as f:
        jobs_json = json.load(f)

    # Check default ENV variable not available outside of step context
    for k in ForwardModelStep.default_env:
        assert k not in os.environ

    for msg in list(ForwardModelRunner(jobs_json).run([])):
        if isinstance(msg, Exited):
            assert msg.exit_code == 0

    # Check default ENV variable not available outside of step context
    for k in ForwardModelStep.default_env:
        assert k not in os.environ
