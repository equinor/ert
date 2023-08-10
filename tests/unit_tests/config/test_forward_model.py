import os
import os.path
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig

from .config_dict_generator import config_generators


@given(config_generators())
def test_ert_config_throws_on_missing_forward_model_job(
    tmp_path_factory, config_generator
):
    with config_generator(tmp_path_factory) as config_values:
        config_values.install_job = []
        config_values.install_job_directory = []
        config_values.forward_model.append(
            ["this-is-not-the-job-you-are-looking-for", "<WAVE-HAND>=casually"]
        )

        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            _ = ErtConfig.from_dict(
                config_values.to_config_dict("test.ert", os.getcwd())
            )


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

    res_config = ErtConfig.from_file(test_config_file_name)
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
    str_with_quotes = """smt/<foo>"/bar"/xx/"t--s.s"/yy/"z/z"/oo"""
    test_config_contents = dedent(
        f"""
        NUM_REALIZATIONS  1
        JOBNAME job_%d
        FORWARD_MODEL COPY_FILE(<FROM>=foo,<TO>={str_with_quotes})
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    ert_config = ErtConfig.from_file(test_config_file_name)
    assert list(ert_config.forward_model_list[0].private_args.values()) == [
        "foo",
        "smt/<foo>/bar/xx/t--s.s/yy/z/z/oo",
    ]


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

    res_config = ErtConfig.from_file(test_config_file_name)
    assert res_config.model_config.jobname_format_string == "job_<IENS>--hei"
    assert (
        res_config.forward_model_list[0].private_args["<TO>"]
        == "something/hello--there.txt"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_quotations_in_forward_model_arglist_are_handled_correctly():
    """This is a regression test, making sure that quoted strings behave consistently.
    They should all result in the same.
    See https://github.com/equinor/ert/issues/2766"""

    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
    NUM_REALIZATIONS  1
    FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>="some stuff", <FILE>=file.txt)
    FORWARD_MODEL COPY_FILE(<FROM>='some, thing', <TO>='some stuff', <FILE>=file.txt)
    FORWARD_MODEL COPY_FILE(<FROM>="some, thing", <TO>="some stuff", <FILE>=file.txt)
    """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ErtConfig.from_file(test_config_file_name)

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
def test_that_positional_forward_model_args_gives_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        FORWARD_MODEL RMS <IENS>
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="Did not expect character: <"):
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


@pytest.mark.usefixtures("use_tmpdir")
def test_that_forward_model_with_different_token_kinds_are_added():
    """
    This is a regression tests for a problem where the parser had different
    token kinds which ended up in separate keys in the input dictionary, and were
    therefore not added
    """
    test_config_file_name = "test.ert"
    Path("job").write_text("EXECUTABLE echo\n", encoding="utf-8")
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS 1
        INSTALL_JOB job job
        FORWARD_MODEL job
        FORWARD_MODEL job(<MESSAGE>=HELLO)
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    assert [
        (j.name, len(j.private_args))
        for j in ErtConfig.from_file(test_config_file_name).forward_model_list
    ] == [("job", 0), ("job", 1)]
