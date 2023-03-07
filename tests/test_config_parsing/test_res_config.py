# pylint: disable=too-many-lines
import logging
import os
import os.path
import re
from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import assume, given

from ert._c_wrappers.config.config_parser import ConfigValidationError, ConfigWarning
from ert._c_wrappers.enkf import ErtConfig
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


@given(config_generators())
def test_ert_config_throws_on_missing_forward_model_job(
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
            ErtConfig.from_file(filename)
        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            ErtConfig.from_dict(config_dict)


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
        match="QUEUE_OPTION MAX_RUNNING is negative",
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
        match="QUEUE_OPTION MAX_RUNNING is negative",
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
def test_parsing_forward_model_with_double_dash_is_possible():
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""

    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        JOBNAME job_%d--hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ErtConfig.from_file(test_config_file_name, use_new_parser=True)
    assert res_config.model_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_list[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


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

    ert_config = ErtConfig.from_file(test_config_file_name, use_new_parser=False)
    for _, value in ert_config.forward_model_list[0].private_args:
        assert " " not in value


@pytest.mark.usefixtures("use_tmpdir")
def test_that_comments_are_ignored():
    """This is a regression test, making sure that we can put double dashes in strings.
    The use case is that a file name is utilized that contains two consecutive hyphens,
    which by the ert config parser used to be interpreted as a comment. In the new
    parser this is allowed"""

    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        --comment
        JOBNAME job_%d--hei --hei
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>=something/hello--there.txt)--foo
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ErtConfig.from_file(test_config_file_name, use_new_parser=True)
    assert res_config.model_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_list[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_quotations_in_forward_model_arglist_are_handled_correctly():
    """This is a regression test, making sure that string behave consistently
    The previous fail cases are described in the comments of the config. They
     should all result in the same
     See https://github.com/equinor/ert/issues/2766"""

    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>="some stuff", <FILE>=file.txt) -- success
        FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>='some stuff', <FILE>=file.txt) -- some stuff becomes somestuff
        FORWARD_MODEL COPY_FILE(<FROM>="some, thing", <TO>="some stuff", <FILE>=file.txt) -- util abort
        """  # noqa: E501
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ErtConfig.from_file(test_config_file_name, use_new_parser=True)

    assert res_config.forward_model_list[0].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_list[0].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_list[0].private_args["<FILE>"] == "file.txt"

    assert res_config.forward_model_list[1].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_list[1].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_list[1].private_args["<FILE>"] == "file.txt"

    assert res_config.forward_model_list[2].private_args["<FROM>"] == "some, thing"
    assert res_config.forward_model_list[2].private_args["<TO>"] == "some stuff"
    assert res_config.forward_model_list[2].private_args["<FILE>"] == "file.txt"


@pytest.mark.usefixtures("use_tmpdir")
def test_parsing_forward_model_with_quotes_in_unquoted_string_fails():
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

    with pytest.raises(ConfigValidationError, match="Expected one of"):
        _ = ErtConfig.from_file(test_config_file_name, use_new_parser=True)


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
        ErtConfig.from_file(test_config_file_name)
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
        ErtConfig.from_file(test_config_file_name)
        assert len(caplog.records) == 1
        assert re.search(r"unresolved arguments.*<FROM>", caplog.messages[0])


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
        ErtConfig.from_file(test_config_file_name)
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
        ErtConfig.from_file(test_config_file_name)


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
        ErtConfig.from_file(test_config_file_name)
        assert len(caplog.records) == 1
        assert re.search(
            r"were not found in the argument list.*<RMS_VERSJON>=2.1",
            caplog.messages[0],
        )


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
        _ = ErtConfig.from_file(test_config_file_name)


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
        match="Cannot setup hook for non-existing job name 'NO_SUCH_JOB'",
    ):
        _ = ErtConfig.from_file(test_config_file_name)


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

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert len(ert_config.forward_model_list) == 1
    job = ert_config.forward_model_list[0]
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
def test_that_installing_two_forward_model_jobs_with_the_same_name_warn():
    test_config_file_name = "test.ert"
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB job job
        INSTALL_JOB job job
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Duplicate forward model job"):
        _ = ErtConfig.from_file(test_config_file_name)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_installing_two_forward_model_jobs_with_the_same_name_warn_with_dir():
    test_config_file_name = "test.ert"
    os.mkdir("jobs")
    Path("jobs/job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB_DIRECTORY jobs
        INSTALL_JOB job job
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.warns(ConfigWarning, match="Duplicate forward model job"):
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
def test_that_spaces_in_forward_model_args_are_dropped():
    test_config_file_name = "test.ert"
    # Intentionally inserted several spaces before comma
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL ECLIPSE100(<VERSION>=smersion                    , <NUM_CPU>=42)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert len(ert_config.forward_model_list) == 1
    job = ert_config.forward_model_list[0]
    assert job.private_args.get("<VERSION>") == "smersion"


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
