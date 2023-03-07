# pylint: disable=too-many-lines
import logging
import os
import os.path
import re
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given

from ert._c_wrappers.config.config_parser import ConfigValidationError, ConfigWarning
from ert._c_wrappers.enkf import ErtConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys

from .config_dict_generator import config_generators, to_config_file


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
        assert len(caplog.records) == 0


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
