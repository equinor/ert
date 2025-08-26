import os
from textwrap import dedent

import pytest

from ert.config.parsing import (
    ConfigValidationError,
    init_user_config_schema,
    parse,
    parse_contents,
)


def touch(filename):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(" ")


def test_that_no_arguments_to_summary_raises_config_validation_error():
    """
    Check that the summary without arglist causes a ConfigValidationError and
    not an error from appending to None parsed from SUMMARY w/o arglist
    """
    with pytest.raises(ConfigValidationError, match="must have at least"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS 1
            SUMMARY
            SUMMARY B 2
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


def test_that_setenv_does_not_expand_envvar():
    config = parse_contents(
        """
        NUM_REALIZATIONS 1
        SETENV PATH $PATH:added
        """,
        file_name="config.ert",
        schema=init_user_config_schema(),
    )
    assert config["SETENV"] == [["PATH", "$PATH:added"]]


def test_that_realisation_is_a_alias_of_realization():
    config = parse_contents(
        "NUM_REALIZATIONS 1", file_name="config.ert", schema=init_user_config_schema()
    )
    assert config["NUM_REALIZATIONS"] == 1


def test_that_new_line_can_be_escaped():
    config = parse_contents(
        """
        NUM_REALIZATIONS \
                1
        """,
        file_name="config.ert",
        schema=init_user_config_schema(),
    )
    assert config["NUM_REALIZATIONS"] == 1


@pytest.mark.filterwarnings(
    "ignore:.*Using DEFINE with substitution strings that are not of "
    "the form '<KEY>'.*:ert.config.ConfigWarning"
)
def test_that_redefines_overwrite_existing_defines():
    config_dict = parse_contents(
        """
        NUM_REALIZATIONS  1
        DEFINE <A> 2
        DEFINE <B> 5
        FORWARD_MODEL MAKE_SYMLINK( \\
                <U>=<A>, \\
                <F>=<B>, \\
                <E>=<O>, \\
                R=Hello, \\
                <R>=R \\
        )
        DEFINE <B> 10
        DEFINE B <A>
        DEFINE D <A>
        DEFINE <A> 3
        FORWARD_MODEL MAKE_SYMLINK(<U>=<A>, <D>=B)
        DEFINE B <A>
        DEFINE C <A>
        """,
        file_name="config.ert",
        schema=init_user_config_schema(),
    )

    defines = config_dict["DEFINE"]
    forward_model_steps = config_dict["FORWARD_MODEL"]

    assert ("<U>", "2") in forward_model_steps[0][1]  # first MAKE_SYMLINK
    assert ("<F>", "5") in forward_model_steps[0][1]  # first MAKE_SYMLINK
    assert ("<U>", "3") in forward_model_steps[1][1]  # second MAKE_SYMLINK
    assert ("<D>", "2") in forward_model_steps[1][1]  # second MAKE_SYMLINK

    assert ["<A>", "2"] not in defines
    assert ["<B>", "2"] not in defines
    assert ["<A>", "3"] in defines
    assert ["D", "2"] in defines
    assert ["B", "3"] in defines
    assert ["C", "3"] in defines


def test_that_including_a_non_existing_file_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match=r"No such file or directory"):
        parse_contents(
            """
            JOBNAME my_name%d
            NUM_REALIZATIONS 1
            INCLUDE does_not_exists
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


def test_that_parsing_a_non_existing_file_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _ = parse("this/is/not/a/file", schema=init_user_config_schema())


def test_that_unknown_queue_option_gives_error_message():
    with pytest.raises(
        ConfigValidationError, match="'QUEUE_OPTION' argument 1 must be one of"
    ):
        _ = parse_contents(
            "QUEUE_OPTION UNKNOWN_QUEUE unsetoption",
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_cyclical_includes_raise_config_validation_error():
    test_config_file_name = "test.ert"

    test_config_self_include = "NUM_REALIZATIONS  1\nINCLUDE test.ert\n"
    test_config_contents = "NUM_REALIZATIONS  1\nINCLUDE include.ert\n"

    test_include_file_name = "include.ert"
    test_include_contents = "JOBNAME included\nINCLUDE test.ert\n"

    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    with open(test_include_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_include_contents)

    with pytest.raises(ConfigValidationError, match=r"Cyclical .*test.ert"):
        _ = parse(test_config_file_name, schema=init_user_config_schema())

    # Test self include raises cyclical include error
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_self_include)
    with pytest.raises(ConfigValidationError, match=r"Cyclical .*test.ert->test.ert"):
        _ = parse(test_config_file_name, schema=init_user_config_schema())


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_no_keywords_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="must be set"):
        _ = parse_contents("", file_name="config.ert", schema=init_user_config_schema())


def test_num_realizations_required_in_config_file():
    with pytest.raises(ConfigValidationError, match=r"NUM_REALIZATIONS must be set.*"):
        _ = parse_contents(
            "ENSPATH storage", file_name="config.ert", schema=init_user_config_schema()
        )


def test_that_invalid_boolean_values_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="boolean"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS  1
            STOP_LONG_RUNNING NOT_YES
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


@pytest.mark.parametrize(
    "config_file_contents",
    [
        dedent(
            """\
             NUM_REALIZATIONS 1
             RUN_TEMPLATE file.txt out.txt
             """
        ),
        dedent(
            """\
             NUM_REALIZATIONS 1
             DATA_FILE file.txt
             """
        ),
        dedent(
            """\
             NUM_REALIZATIONS 1
             GEN_KW file.txt file.txt
             """
        ),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_file_without_read_access_raises_config_validation_error(
    config_file_contents,
):
    """Given a file without read permissions, we should fail gracefully"""

    config_file_name = "config.ert"
    template_file = "file.txt"

    touch(template_file)
    os.chmod(template_file, 0o000)  # Remove permissions
    with pytest.raises(
        ConfigValidationError,
        match=f'{template_file}" is not readable; please check read access.',
    ):
        with open(config_file_name, mode="w", encoding="utf-8") as file:
            file.write(config_file_contents)

        _ = parse(config_file_name, schema=init_user_config_schema())


@pytest.mark.usefixtures("use_tmpdir")
def test_not_executable_job_script_raises_config_validation_error():
    """Given a non executable job script, we should fail gracefully"""

    config_file_name = "config.ert"
    script_name = "not-executable-script.py"
    touch(script_name)
    config_file_contents = dedent(
        f"""\
         NUM_REALIZATIONS 1
         JOB_SCRIPT {script_name}
         """
    )
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(ConfigValidationError, match=f"not executable.*{script_name}"):
        _ = parse(config_file_name, schema=init_user_config_schema())


@pytest.mark.usefixtures("use_tmpdir")
def test_not_executable_job_script_somewhere_in_PATH_raises_config_validation_error(
    monkeypatch,
):
    """Given a non executable job script referred to by relative path (in this case:
    just the filename) in the config file, where the relative path is not relative to
    the location of the config file / current directory, but rather some location
    specified by the env var PATH, we should fail gracefully"""

    config_file_name = "config.ert"
    script_name = "not-executable-script.py"
    path_location = os.path.join(os.getcwd(), "bin")
    os.mkdir(path_location)
    touch(os.path.join(path_location, script_name))
    os.chmod(path_location, 0o000)
    monkeypatch.setenv("PATH", path_location, ":")
    config_file_contents = dedent(
        f"""\
         NUM_REALIZATIONS 1
         JOB_SCRIPT {script_name}
         """
    )
    with open(config_file_name, mode="w", encoding="utf-8") as fh:
        fh.write(config_file_contents)
    with pytest.raises(
        ConfigValidationError,
        match="Could not find executable",
    ):
        _ = parse(config_file_name, schema=init_user_config_schema())

    os.chmod(path_location, 0o775)


def test_that_giving_non_int_values_in_num_realization_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="integer"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS hello
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


def test_that_giving_non_float_values_in_enkf_alpha_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="number"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS  1
            ENKF_ALPHA  hello
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_giving_non_executable_in_job_script_raises_config_validation_error():
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """\
        NUM_REALIZATIONS  1
        JOB_SCRIPT  not-an-executable-anyone-would-have-on-their-laptop
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    with pytest.raises(ConfigValidationError, match="executable"):
        _ = parse(test_config_file_name, schema=init_user_config_schema())


def test_that_giving_too_many_arguments_to_enkf_alpha_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="maximum 1 arguments"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS  1
            ENKF_ALPHA 1.0 2.0 3.0
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )


def test_that_giving_too_few_arguments_to_enkf_alpha_raises_config_validation_error():
    with pytest.raises(ConfigValidationError, match="at least 1 arguments"):
        _ = parse_contents(
            """
            NUM_REALIZATIONS  1
            ENKF_ALPHA
            """,
            file_name="config.ert",
            schema=init_user_config_schema(),
        )
