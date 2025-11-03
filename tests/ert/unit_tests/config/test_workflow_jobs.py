from pathlib import Path
from textwrap import dedent

import pytest

from ert import ErtScript
from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigValidationError, ConfigWarning, ErtConfig, Workflow
from ert.config.workflow_job import (
    ErtScriptWorkflow,
    UserInstalledErtScriptWorkflow,
    workflow_job_from_file,
)
from ert.plugins import ErtRuntimePlugins, get_site_plugins
from ert.storage import Storage


def test_reading_non_existent_workflow_job_raises_config_error():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        workflow_job_from_file("/tmp/does_not_exist", origin="user")


def test_that_ert_warns_on_duplicate_workflow_jobs(tmp_path):
    """
    Tests that we emit a ConfigWarning if we detect multiple
    workflows with the same name during config parsing.
    Relies on the internal workflow CAREFUL_COPY_FILE.
    """
    test_workflow_job = tmp_path / "CAREFUL_COPY_FILE"
    Path(test_workflow_job).write_text(
        "EXECUTABLE test_copy_duplicate.py", encoding="utf-8"
    )
    test_workflow_job_executable = tmp_path / "test_copy_duplicate.py"
    Path(test_workflow_job_executable).touch(mode=0o755)
    test_config_file_name = tmp_path / "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        LOAD_WORKFLOW_JOB CAREFUL_COPY_FILE
        """
    )
    Path(test_config_file_name).write_text(test_config_contents, encoding="utf-8")

    with (
        pytest.warns(
            ConfigWarning, match="Duplicate workflow jobs with name 'CAREFUL_COPY_FILE'"
        ),
    ):
        _ = ErtConfig.with_plugins(get_site_plugins()).from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_stop_on_fail_is_parsed_external():
    Path("fail_job").write_text(
        "EXECUTABLE echo\nMIN_ARG 1\nSTOP_ON_FAIL True\n", encoding="utf-8"
    )

    job_internal = workflow_job_from_file(
        name="FAIL", config_file="fail_job", origin="user"
    )

    assert job_internal.stop_on_fail


def test_that_site_ertscript_is_serialized_to_name_and_deserialized_from_plugins():
    class SomeScript(ErtScript):
        def run(self, storage: Storage):
            pass

    workflow_job = ErtScriptWorkflow(
        name="SOME_SITE_INSTALLED_WFJOB",
        ert_script=SomeScript,
    )

    with use_runtime_plugins(
        ErtRuntimePlugins(
            installed_workflow_jobs={
                "SOME_SITE_INSTALLED_WFJOB": ErtScriptWorkflow(
                    name="SOME_SITE_INSTALLED_WFJOB",
                    ert_script=SomeScript,
                ),
            }
        )
    ):
        workflow_job_json = workflow_job.model_dump(mode="json", exclude_unset=True)
        assert workflow_job_json == {
            "name": "SOME_SITE_INSTALLED_WFJOB",
            "type": "site_installed",
        }

        assert (
            ErtScriptWorkflow.model_validate(workflow_job_json).ert_script == SomeScript
        )

    class RevisedErtScript(ErtScript):
        def run(self, storage: Storage):
            pass

    with use_runtime_plugins(
        ErtRuntimePlugins(
            installed_workflow_jobs={
                "SOME_SITE_INSTALLED_WFJOB": ErtScriptWorkflow(
                    name="SOME_SITE_INSTALLED_WFJOB",
                    ert_script=RevisedErtScript,
                ),
            }
        )
    ):
        # Expect it to lookup revised ertscript from site plugins via name
        assert (
            ErtScriptWorkflow.model_validate(workflow_job_json).ert_script
            == RevisedErtScript
        )


def test_that_user_installed_ertscript_serializes_as_source_and_loads_class_on_demand(
    use_tmpdir,
):
    Path("script.py").write_text(
        """
from ert import ErtScript
from ert.storage import Storage

class SomeScript(ErtScript):
    def run(self, storage: Storage):
        return 'i am here'
    """,
        encoding="utf-8",
    )

    wfjob = UserInstalledErtScriptWorkflow(
        name="Script1",
        source="script.py",
    )

    script_class = wfjob.load_ert_script_class()
    script_instance = script_class()
    assert script_instance.run(storage=None) == "i am here"

    # Expect only reference stored, i.e., name
    serialized_json = wfjob.model_dump(mode="json", exclude_unset=True)
    assert serialized_json == {
        "name": "Script1",
        "source": "script.py",
    }

    # Expect it to have the class resolved again
    assert (
        UserInstalledErtScriptWorkflow.model_validate(serialized_json)
        .load_ert_script_class()()
        .run(storage=None)
        == "i am here"
    )


def test_that_site_and_user_installed_fm_steps_are_serialized_differently(use_tmpdir):
    Path("USER_EXECUTABLE_JOB_FILE").write_text(
        "EXECUTABLE echo",
        encoding="utf-8",
    )

    Path("script.py").write_text(
        """
from ert import ErtScript
from ert.storage import Storage

class SomeScript(ErtScript):
    def run(self, storage: Storage):
        return 'i am here'
    """,
        encoding="utf-8",
    )

    Path("USER_ERTSCRIPT_JOB_FILE").write_text(
        """
        INTERNAL True
        SCRIPT script.py
""",
        encoding="utf-8",
    )

    Path("DUMMY_WORKFLOW_FILE").write_text(
        """
        USR_INSTALLED_EXECUTABLE_JOB
        SITE_INSTALLED_JOB
        USR_INSTALLED_ERTSCRIPT_JOB
    """,
        encoding="utf-8",
    )

    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        LOAD_WORKFLOW_JOB USER_EXECUTABLE_JOB_FILE USR_INSTALLED_EXECUTABLE_JOB
        LOAD_WORKFLOW_JOB USER_ERTSCRIPT_JOB_FILE USR_INSTALLED_ERTSCRIPT_JOB
        LOAD_WORKFLOW DUMMY_WORKFLOW_FILE DUMMY_WORKFLOW
        HOOK_WORKFLOW DUMMY_WORKFLOW POST_EXPERIMENT
        """
    )
    Path("config.ert").write_text(test_config_contents, encoding="utf-8")

    class SiteWorkflowJobScript(ErtScript):
        def run(self, storage: Storage):
            pass

    site_plugins = ErtRuntimePlugins(
        installed_workflow_jobs={
            "SITE_INSTALLED_JOB": ErtScriptWorkflow(
                name="SITE_INSTALLED_JOB",
                ert_script=SiteWorkflowJobScript,
            )
        }
    )
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")
    workflow = ert_config.workflows["DUMMY_WORKFLOW"]
    workflow_jobs = [wfj[0] for wfj in workflow.cmd_list]
    assert [job.type for job in workflow_jobs] == [
        "user_installed_executable",
        "site_installed",
        "user_installed_ertscript",
    ]
    [user_executable_wf, site_wf, user_ertscript_wf] = workflow_jobs
    assert user_executable_wf.name == "USR_INSTALLED_EXECUTABLE_JOB"
    assert user_executable_wf.executable == "echo"

    assert site_wf.name == "SITE_INSTALLED_JOB"
    assert site_wf.ert_script == SiteWorkflowJobScript

    assert user_ertscript_wf.name == "USR_INSTALLED_ERTSCRIPT_JOB"
    assert user_ertscript_wf.source == str(Path("script.py").resolve())

    with use_runtime_plugins(site_plugins):
        assert Workflow(**workflow.model_dump(mode="json")) == workflow
