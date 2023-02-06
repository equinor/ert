import logging
import os
import re
from datetime import date
from textwrap import dedent

import pytest
from hypothesis import assume, given

from ert._c_wrappers.config.config_parser import ConfigValidationError, ConfigWarning
from ert._c_wrappers.enkf import ResConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys

from .config_dict_generator import config_generators, to_config_file


def touch(filename):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(" ")


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")

    rconfig = None
    with pytest.raises(
        ConfigValidationError, match=r"Parsing.*resulted in the errors:"
    ):
        rconfig = ResConfig(user_config_file=str(tmp_path / "test.ert"))

    assert rconfig is None


def test_num_realizations_required_in_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file_name = "config.ert"
    config_file_contents = "ENSPATH storage"
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(ConfigValidationError, match=r"NUM_REALIZATIONS must be set.*"):
        ResConfig(user_config_file=config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_res_config_parses_date():
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
    res_config = ResConfig(user_config_file=test_config_file_name)

    date_string = date.today().isoformat()
    expected_storage = os.path.abspath(f"storage/{test_config_file_base}-{date_string}")
    expected_run_path = f"{expected_storage}/runpath/realization-<IENS>/iter-<ITER>"
    expected_ens_path = f"{expected_storage}/ensemble"
    assert res_config.ens_path == expected_ens_path
    assert res_config.model_config.runpath_format_string == expected_run_path


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_that_creating_res_config_from_dict_is_same_as_from_file(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        assert ResConfig(config_dict=config_dict) == ResConfig(filename)


@given(config_generators())
def test_res_config_throws_on_missing_forward_model_job(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory) as config_dict:
        config_dict.pop(ConfigKeys.INSTALL_JOB)
        config_dict.pop(ConfigKeys.INSTALL_JOB_DIRECTORY)
        config_dict[ConfigKeys.FORWARD_MODEL].append(
            ["this-is-not-the-job-you-are-looking-for", "<WAVE-HAND>=casually"]
        )

        to_config_file(filename, config_dict)

        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            ResConfig(user_config_file=filename)
        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            ResConfig(config_dict=config_dict)


def test_default_ens_path(tmpdir):
    with tmpdir.as_cwd():
        config_file = "test.ert"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(
                """
NUM_REALIZATIONS  1
            """
            )
        res_config = ResConfig(config_file)
        # By default, the ensemble path is set to 'storage'
        default_ens_path = res_config.ens_path

        with open(config_file, "a", encoding="utf-8") as f:
            f.write(
                """
ENSPATH storage
            """
            )

        # Set the ENSPATH in the config file
        res_config = ResConfig(config_file)
        set_in_file_ens_path = res_config.ens_path

        assert default_ens_path == set_in_file_ens_path

        config_dict = {
            ConfigKeys.NUM_REALIZATIONS: 1,
            "ENSPATH": os.path.join(os.getcwd(), "storage"),
        }

        dict_set_ens_path = ResConfig(config_dict=config_dict).ens_path

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
        match="QUEUE_OPTION MAX_RUNNING is negative",
    ):
        ResConfig(user_config_file=test_config_file_name)


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
        match="QUEUE_OPTION MAX_RUNNING is negative",
    ):
        ResConfig(config_dict=config_dict)


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
        ResConfig(user_config_file=test_config_file_name)


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
        _ = ResConfig(user_config_file=test_config_file_name)


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
        match="Cannot find file or directory does_not_exist",
    ):
        ResConfig(user_config_file=test_config_file_name)


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
        match="Cannot find file or directory does_not_exist",
    ):
        ResConfig(user_config_file=test_config_file_name)


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
        _ = ResConfig(user_config_file=test_config_file_name)


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
        ResConfig(user_config_file=test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_parsing_forward_model_with_quotes_does_not_introduce_spaces():
    """this is a regression test, making sure that we do not by mistake introduce
    spaces while parsing forward model lines that contain quotation marks

    the use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser is interpreted as a comment - to circumvent the
    comment interpretation, quotation marks are used"""

    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME job_%d
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/"hello--there.txt")
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ResConfig(user_config_file=test_config_file_name)
    for _, value in res_config.forward_model.jobs[0].private_args:
        assert " " not in value


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_with_all_args_resolved_gives_no_warning(caplog):
    # given a forward model that references a job
    # and that job has args in its arglist that require substitution
    # and the defines or the private args or defaults resolve all substitutions
    # then we should have no warning

    test_config_file_name = "test.ert"
    # We're using the job old-style/COPY_FILE, included through site-config, and rely
    # on it having two arguments, TO and FROM
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<FROM>=bar,<TO>=foo)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with caplog.at_level(logging.WARNING):
        ResConfig(user_config_file=test_config_file_name)
        assert len(caplog.records) == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_with_unsubstituted_arg_gives_warning(caplog):
    # given a forward model that references a job
    # and that job has args in its arglist that require substitution
    # and the defines and defaults and the private args do not resolve all args
    # then we should have a warning

    test_config_file_name = "test.ert"
    # We're using the job old-style/COPY_FILE, included through site-config, and rely
    # on it having two arguments, TO and FROM
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<TO>=foo, <FROM>=some-<BLA>)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with caplog.at_level(logging.WARNING):
        ResConfig(user_config_file=test_config_file_name)
        assert len(caplog.records) == 1
        assert re.search(
            r"unresolved arguments.*<FROM>.*some-<BLA>", caplog.messages[0]
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_with_unresolved_IENS_or_ITER_gives_no_warning(caplog):
    # given a forward model that references a job
    # and that job has args in its arglist that require substitution
    # and the defines and defaults and the private args do not resolve all args
    # and the unresolved strings are <ITER> or <IENS>
    # then we should have no warning
    test_config_file_name = "test.ert"
    # We're using the job old-style/COPY_FILE, included through site-config, and rely
    # on it having two arguments, TO and FROM
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<FROM>=something-<ITER>, <TO>=foo)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with caplog.at_level(logging.WARNING):
        ResConfig(user_config_file=test_config_file_name)
        assert len(caplog.records) == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_with_resolved_substitutions_by_default_values_gives_no_error(
    caplog,
):
    # given a forward model that references a job
    # and that job has args in its arglist that require substitution
    # and the job has default values for some arguments
    # and defines and private args cover the arguments without default value
    # then we should have no warning
    test_config_file_name = "test.ert"
    # We're using the job old-style/RMS, included through site-config, which has a
    # bunch of args, and some default values
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <RMS_PROJECT> spam
        DEFINE <RMS_WORKFLOW> frying
        DEFINE <RMS_TARGET_FILE> result
        FORWARD_MODEL RMS(<IENS>=2, <RMS_VERSION>=2.1)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with caplog.at_level(logging.WARNING):
        ResConfig(user_config_file=test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_forward_model_warns_for_private_arg_without_effect(caplog):
    # given a forward model that references a job
    # and there is a key=value private arg given in the forward model
    # and that substitution has no effect on the arglist
    # then we should have a warning
    test_config_file_name = "test.ert"
    # We're using the job old-style/RMS, included through site-config, which has a
    # bunch of args. use case here is a typo
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <RMS_VERSION> 2.4
        DEFINE <RMS_PROJECT> spam
        DEFINE <RMS_WORKFLOW> frying
        DEFINE <RMS_TARGET_FILE> result
        FORWARD_MODEL RMS(<IENS>=<FOO>-2,<RMS_VERSJON>=2.1)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with caplog.at_level(logging.WARNING):
        ResConfig(user_config_file=test_config_file_name)
        assert len(caplog.records) == 1
        assert re.search(r"no effect.*<RMS_VERSJON>=2.1", caplog.messages[0])


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

    res_config = ResConfig("config.ert")

    assert res_config.workflows["workflow"].cmd_list[0][1] == ["0"]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_positional_forward_model_args_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL RMS(<IENS>)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="FORWARD_MODEL RMS"):
        _ = ResConfig(user_config_file=test_config_file_name)


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

    with pytest.raises(ConfigValidationError, match="Could not find job `NO_SUCH_JOB`"):
        _ = ResConfig(user_config_file=test_config_file_name)


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
        _ = ResConfig(user_config_file=test_config_file_name)


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
        match="Cannot setup hook for non-existing job name NO_SUCH_JOB",
    ):
        _ = ResConfig(user_config_file=test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_substitutions_can_be_done_in_job_names():
    """
    Regression test for a usage case involving setting ECL100 or ECL300
    that was broken by changes to forward_model substitutions.
    """
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <ECL100OR300> E100
        FORWARD_MODEL ECLIPS<ECL100OR300>(<VERSION>=1, <NUM_CPU>=42, <OPTS>="-m")
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ResConfig(user_config_file=test_config_file_name)
    assert len(res_config.forward_model.jobs) == 1
    job = res_config.forward_model.jobs[0]
    assert job.name == "ECLIPSE100"


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
            _ = ResConfig(config_dict=config_dict)
