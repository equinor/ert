import pytest

from ert._c_wrappers.enkf import HookRuntime, ResConfig


def test_that_unknown_queue_option_gives_error_message(tmp_path, capsys):
    # AS AN ert user

    # WHEN my user config file contains an incorrect queue option
    test_user_config = tmp_path / "user_config.ert"
    test_user_config.write_text(
        "JOBNAME  Job%d\nRUNPATH /tmp/simulations/run%d\n"
        "NUM_REALIZATIONS 10\nQUEUE_OPTION UNKNOWN_QUEUE unsetoption\n"
    )

    # THEN I expect to get a good error message
    with pytest.raises(ValueError, match="Parsing"):
        _ = ResConfig(str(test_user_config))
    err = capsys.readouterr().err
    assert "Errors parsing" in err
    assert "UNKNOWN_QUEUE" in err


@pytest.mark.parametrize(
    "run_mode",
    [
        HookRuntime.POST_SIMULATION,
        HookRuntime.PRE_SIMULATION,
        HookRuntime.PRE_FIRST_UPDATE,
        HookRuntime.PRE_UPDATE,
        HookRuntime.POST_UPDATE,
    ],
)
def test_that_workflow_run_modes_can_be_selected(tmp_path, run_mode):
    # AS AN ert user

    # GIVEN A workflow file
    my_script = (tmp_path / "MY_WORKFLOW").resolve()
    my_script.write_text("EXPORT_RUNPATH *")

    # WHEN I select a run mode for my workflow
    test_user_config = tmp_path / "user_config.ert"
    test_user_config.write_text(
        "NUM_REALIZATIONS 10\n"
        f"LOAD_WORKFLOW {my_script} SCRIPT\n"
        f"HOOK_WORKFLOW SCRIPT {run_mode}\n"
    )

    # THEN I expect that the workflow runs at the selected time
    res_config = ResConfig(str(test_user_config))
    assert (
        len(list(res_config.ert_workflow_list.get_workflows_hooked_at(run_mode))) == 1
    )
