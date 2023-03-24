# pylint: disable=too-many-lines
import logging
import os
import os.path
from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import assume, given

from ert._c_wrappers.config.config_parser import (
    CombinedConfigError,
    ConfigValidationError,
    ConfigWarning,
)
from ert._c_wrappers.enkf import ErtConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.util import SubstitutionList

from .config_dict_generator import config_generators


def touch(filename):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(" ")


def test_add_combined_config_error_to_combined_config_error():
    dummy_src_combined_error = CombinedConfigError(
        [
            ConfigValidationError(errors="1"),
            ConfigValidationError(errors="2"),
            ConfigValidationError(errors="3"),
        ]
    )

    dummy_dest_combined_error = CombinedConfigError(
        [ConfigValidationError(errors="ZERO")]
    )

    dummy_dest_combined_error.add_error(dummy_src_combined_error)
    expected_str = "".join(str(x) for x in dummy_dest_combined_error.errors)

    assert expected_str == "ZERO123"


def test_add_config_error_to_combined_config_error():
    combined_error = CombinedConfigError()
    combined_error.add_error(ConfigValidationError("123"))

    assert str(combined_error) == "123"


def test_add_combined_config_error_to_itself_raises_error():
    combined_error = CombinedConfigError()
    combined_error.add_error(ConfigValidationError("123"))

    with pytest.raises(ValueError, match=".* itself"):
        combined_error.add_error(combined_error)


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")

    rconfig = None
    with pytest.raises(
        ConfigValidationError, match=r"Parsing.*resulted in the errors:"
    ):
        rconfig = ErtConfig.from_file(str(tmp_path / "test.ert"), use_new_parser=False)

    assert rconfig is None


def test_num_realizations_required_in_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file_name = "config.ert"
    config_file_contents = "ENSPATH storage"
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(ConfigValidationError, match=r"NUM_REALIZATIONS must be set.*"):
        ErtConfig.from_file(config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_config_parses_date():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    ert_config = ErtConfig.from_file(test_config_file_name)

    date_string = date.today().isoformat()
    expected_storage = os.path.abspath(f"storage/{test_config_file_base}-{date_string}")
    expected_run_path = f"{expected_storage}/runpath/realization-<IENS>/iter-<ITER>"
    expected_ens_path = f"{expected_storage}/ensemble"
    assert ert_config.ens_path == expected_ens_path
    assert ert_config.model_config.runpath_format_string == expected_run_path


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_that_creating_ert_config_from_dict_is_same_as_from_file(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        assert ErtConfig.from_dict(config_dict) == ErtConfig.from_file(filename)


def test_default_ens_path(tmpdir):
    with tmpdir.as_cwd():
        config_file = "test.ert"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(
                """
NUM_REALIZATIONS  1
            """
            )
        ert_config = ErtConfig.from_file(config_file)
        # By default, the ensemble path is set to 'storage'
        default_ens_path = ert_config.ens_path

        with open(config_file, "a", encoding="utf-8") as f:
            f.write(
                """
ENSPATH storage
            """
            )

        # Set the ENSPATH in the config file
        ert_config = ErtConfig.from_file(config_file)
        set_in_file_ens_path = ert_config.ens_path

        assert default_ens_path == set_in_file_ens_path

        config_dict = {
            ConfigKeys.NUM_REALIZATIONS: 1,
            "ENSPATH": os.path.join(os.getcwd(), "storage"),
        }

        dict_set_ens_path = ErtConfig.from_dict(config_dict).ens_path

        assert dict_set_ens_path == config_dict["ENSPATH"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_queue_config_content_negative_value_invalid():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        QUEUE_SYSTEM LOCAL
        QUEUE_OPTION LOCAL MAX_RUNNING -4
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="QUEUE_OPTION LOCAL MAX_RUNNING is negative",
    ):
        ErtConfig.from_file(test_config_file_name)


@given(config_generators())
def test_that_queue_config_dict_negative_value_invalid(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_dict:
        config_dict[ConfigKeys.QUEUE_OPTION].append(
            ["LSF", "MAX_RUNNING", "-6"],
        )

    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="QUEUE_OPTION LSF MAX_RUNNING is negative",
    ):
        ErtConfig.from_dict(config_dict)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_non_existant_job_directory_gives_config_validation_error():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        INSTALL_JOB_DIRECTORY does_not_exist
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="Unable to locate job directory",
    ):
        ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_empty_job_directory_gives_warning():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        INSTALL_JOB_DIRECTORY empty
        """
    )
    os.mkdir("empty")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.warns(ConfigWarning, match="No files found in job directory"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_loading_non_existant_workflow_gives_validation_error():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        LOAD_WORKFLOW does_not_exist
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match='Cannot find file or directory "does_not_exist" ',
    ):
        ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_loading_non_existant_workflow_job_gives_validation_error():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        LOAD_WORKFLOW_JOB does_not_exist
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match='Cannot find file or directory "does_not_exist"',
    ):
        ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_job_definition_file_with_unexecutable_script_gives_validation_error():
    test_config_file_name = "test.ert"
    job_script_file = os.path.abspath("not_executable")
    job_name = "JOB_NAME"
    test_config_contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        LOAD_WORKFLOW_JOB {job_name}
        """
    )
    with open(job_name, "w", encoding="utf-8") as fh:
        fh.write(f"EXECUTABLE {job_script_file}\n")
    with open(job_script_file, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/bash\n")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.raises(
        expected_exception=ConfigValidationError,
    ):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_a_config_warning_is_given_when_eclbase_and_jobname_is_given():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME job_%d
        ECLBASE base_%d
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with pytest.warns(ConfigWarning):
        ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_magic_strings_get_substituted_in_workflow():
    script_file_contents = dedent(
        """
        SCRIPT script.py
        ARGLIST <A>
        ARG_TYPE 0 INT
        """
    )
    workflow_file_contents = dedent(
        """
        script <ZERO>
        """
    )
    script_file_path = os.path.join(os.getcwd(), "script")
    workflow_file_path = os.path.join(os.getcwd(), "workflow")
    with open(script_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(script_file_contents)
    with open(workflow_file_path, mode="w", encoding="utf-8") as fh:
        fh.write(workflow_file_contents)

    with open("script.py", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                """
                from ert._c_wrappers.job_queue import ErtScript
                class Script(ErtScript):
                    def run(self, *args):
                        pass
                """
            )
        )
    with open("config.ert", mode="w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS 1
                DEFINE <ZERO> 0
                LOAD_WORKFLOW_JOB {script_file_path} script
                LOAD_WORKFLOW {workflow_file_path}
                """
            )
        )

    ert_config = ErtConfig.from_file("config.ert")

    assert ert_config.workflows["workflow"].cmd_list[0][1] == ["0"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unknown_job_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        SIMULATION_JOB NO_SUCH_JOB
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="Could not find job 'NO_SUCH_JOB'"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_both_jobname_and_eclbase_gives_warning():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME job1
        ECLBASE job2
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Can not have both JOBNAME and ECLBASE"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unknown_hooked_job_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        HOOK_WORKFLOW NO_SUCH_JOB PRE_SIMULATION
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(
        ConfigValidationError,
        match=r".* non-existing workflow .* 'NO_SUCH_JOB'",
    ):
        ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unknown_run_mode_gives_config_validation_error():
    content_dict = {
        "HOOK_WORKFLOW": [["MAKE_DIRECTORY", "PRE_SIMULATIONnn"]],
    }

    substitution_list = SubstitutionList.from_dict({})

    with pytest.raises(
        ConfigValidationError,
        match="Run mode .* not supported for Hook Workflow",
    ):
        _ = ErtConfig._workflows_from_dict(content_dict, substitution_list)


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_that_if_field_is_given_and_grid_is_missing_you_get_error(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_dict:
        del config_dict[ConfigKeys.GRID]
        assume(len(config_dict.get(ConfigKeys.FIELD_KEY, [])) > 0)
        with pytest.raises(
            ConfigValidationError,
            match="In order to use the FIELD keyword, a GRID must be supplied",
        ):
            _ = ErtConfig.from_dict(config_dict)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_include_statements_with_multiple_values_raises_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE this and that and some-other
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(
        ConfigValidationError, match="Keyword:INCLUDE must have exactly one argument"
    ):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_workflows_with_errors_are_not_loaded():
    """
    The user may install several workflows with LOAD_WORKFLOW_DIRECTORY that
    does not work with the current versions of plugins installed in the system,
    but could have worked with an older or newer version of the packages installed.

    Therefore the user should be warned about workflows that have issues, and not be
    able to run those later. If a workflow with errors are hooked, then the user will
    get an error indicating that there is no such workflow.
    """
    test_config_file_name = "test.ert"
    Path("WFJOB").write_text("EXECUTABLE echo\n", encoding="utf-8")
    # intentionally misspelled WFJOB as WFJAB
    Path("wf").write_text("WFJAB hello world\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME JOOOOOB
        LOAD_WORKFLOW_JOB WFJOB
        LOAD_WORKFLOW wf
        """
    )

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Encountered error.*while reading workflow"):
        ert_config = ErtConfig.from_file(test_config_file_name)
        assert "wf" not in ert_config.workflows


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "load_statement", ["LOAD_WORKFLOW_JOB wfs/WFJOB", "WORKFLOW_JOB_DIRECTORY wfs"]
)
def test_that_failing_to_load_ert_script_with_errors_fails_gracefully(load_statement):
    """
    The user may install several workflow jobs with LOAD_WORKFLOW_JOB_DIRECTORY that
    does not work with the current versions of plugins installed in the system,
    but could have worked with an older or newer version of the packages installed.

    Therefore the user should be warned about workflow jobs that have issues, and not be
    able to run those later.
    """
    test_config_file_name = "test.ert"
    Path("wfs").mkdir()
    Path("wfs/WFJOB").write_text("SCRIPT wf_script.py\nINTERNAL True", encoding="utf-8")
    Path("wf_script.py").write_text("", encoding="utf-8")
    test_config_contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        {load_statement}
        """
    )

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(
        ConfigWarning, match="Loading workflow job.*failed.*It will not be loaded."
    ):
        ert_config = ErtConfig.from_file(test_config_file_name)
        assert "wf" not in ert_config.workflows


@pytest.mark.usefixtures("use_tmpdir")
def test_that_define_statements_with_less_than_one_argument_raises_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <USER>
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(
        ConfigValidationError, match="Keyword:DEFINE must have two or more"
    ):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_non_int_values_give_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  hello
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="integer"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_non_float_values_give_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        ENKF_ALPHA  hello
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="number"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_non_executable_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOB_SCRIPT  hello
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="[Ee]xecutable"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_too_many_arguments_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        ENKF_ALPHA 1.0 2.0 3.0
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="maximum 1 arguments"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_too_few_arguments_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        ENKF_ALPHA
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="at least 1 arguments"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_include_statements_work():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE include.ert
        """
    )
    test_include_file_name = "include.ert"
    test_include_contents = dedent(
        """
        JOBNAME included
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert ert_config.model_config.jobname_format_string == "included"


@pytest.mark.usefixtures("use_tmpdir")
def test_include_cyclical_raises_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE include.ert
        """
    )
    test_include_file_name = "include.ert"
    test_include_contents = dedent(
        """
        JOBNAME included
        INCLUDE test.ert
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    with pytest.raises(ConfigValidationError, match="Cyclical .*test.ert"):
        ErtConfig.from_file(test_config_file_name, use_new_parser=True)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_incorrect_queue_name_in_queue_option_fails():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        QUEUE_OPTION VOCAL MAX_RUNNING 50
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="VOCAL"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_no_keywords_fails_gracefully():
    test_config_file_name = "test.ert"
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write("")

    with pytest.raises(ConfigValidationError, match="must be set"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_recursive_defines_fails_gracefully_and_logs(caplog):
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        DEFINE <A> 1<A>
        NUM_REALIZATIONS  <A>
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="integer"), caplog.at_level(
        logging.WARNING
    ):
        _ = ErtConfig.from_file(test_config_file_name)
    assert len(caplog.records) == 1
    assert "max iterations" in caplog.records[0].msg


@pytest.mark.usefixtures("use_tmpdir")
def test_that_define_string_quotes_are_removed():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        DEFINE <A> "A"
        NUM_REALIZATIONS 1
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    ert_Config = ErtConfig.from_file(test_config_file_name)
    assert ert_Config.substitution_list.get("<A>") == "A"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_included_files_uses_paths_relative_to_itself():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INCLUDE includes/install_jobs.ert
        """
    )
    os.mkdir("includes")
    test_include_file_name = "includes/install_jobs.ert"
    test_include_contents = dedent(
        """
        INSTALL_JOB FM ../FM
        """
    )
    test_fm_file_name = "FM"
    test_fm_contents = dedent(
        """
        EXECUTABLE echo
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)
    with open(test_fm_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_fm_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert ert_config.installed_jobs["FM"].name == "FM"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_invalid_boolean_values_are_handled_gracefully():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        STOP_LONG_RUNNING NOT_YES
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="boolean"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("val, expected", [("TrUe", True), ("FaLsE", False)])
def test_that_boolean_values_can_be_any_case(val, expected):
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        STOP_LONG_RUNNING {val}
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert ert_config.analysis_config.get_stop_long_running() == expected


@pytest.mark.usefixtures("use_tmpdir")
def test_not_executable_job_script_fails_gracefully():
    """Given a non executable job script, we should fail gracefully"""

    config_file_name = "config.ert"
    script_name = "not-executable-script.py"
    touch(script_name)
    config_file_contents = dedent(
        f"""NUM_REALIZATIONS 1
         JOB_SCRIPT {script_name}
         """
    )
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(ConfigValidationError, match=f"not executable.*{script_name}"):
        ErtConfig.from_file(config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_not_executable_job_script_somewhere_in_PATH_fails_gracefully(monkeypatch):
    """Given a non executable job script referred to by relative path (in this case:
    just the filename) in the config file, where the relative path is not relative to
    the location of the config file / current directory, but rather some location
    specified by the env var PATH, we should fail gracefully"""

    config_file_name = "config.ert"
    script_name = "not-executable-script.py"
    path_location = os.path.join(os.getcwd(), "bin")
    os.mkdir(path_location)
    touch(os.path.join(path_location, script_name))
    os.chmod(path_location, 0x0)
    monkeypatch.setenv("PATH", path_location, ":")
    config_file_contents = dedent(
        f"""NUM_REALIZATIONS 1
         JOB_SCRIPT {script_name}
         """
    )
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(
        ConfigValidationError,
        match="Could not find executable|Executable.*does not exist",
    ):
        ErtConfig.from_file(config_file_name)

    os.chmod(path_location, 0x775)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_include_take_into_account_path():
    """
    Tests that use_new_parser resolves an issue
    with the old parser where the first relative path
    FORWARD_MODEL is chosen for all.
    """
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        INCLUDE dir/include.ert
        INSTALL_JOB job2 job2
        """
    )
    test_include_file_name = "dir/include.ert"
    test_include_contents = dedent(
        """
        INSTALL_JOB job1 job1
        """
    )
    # The old parser tries to find dir/job2
    os.mkdir("dir")
    Path("dir/job1").write_text("EXECUTABLE echo\n", encoding="utf-8")
    Path("job2").write_text("EXECUTABLE ls\n", encoding="utf-8")
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    ert_config = ErtConfig.from_file(test_config_file_name, use_new_parser=True)
    assert list(ert_config.installed_jobs.keys()) == [
        "job1",
        "job2",
    ]
