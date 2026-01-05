import copy
import logging
import os
import os.path
import stat
from pathlib import Path
from textwrap import dedent

import pytest

from ert.config import ErtConfig, ForwardModelStep
from ert.config.ert_config import (
    create_forward_model_json,
    forward_model_step_from_config_contents,
)


@pytest.fixture
def context():
    return {"<RUNPATH>": "./"}


@pytest.fixture
def fm_step_list():
    result = [
        {
            "name": "PERLIN",
            "executable": "perlin.py",
            "target_file": "my_target_file",
            "error_file": "error_file",
            "start_file": "some_start_file",
            "stdout": "perlin.stdout",
            "stderr": "perlin.stderr",
            "stdin": "input4thewin",
            "argList": ["-speed", "hyper"],
            "environment": {"TARGET": "flatland"},
            "max_running_minutes": 12,
        },
        {
            "name": "AGGREGATOR",
            "executable": "aggregator.py",
            "target_file": "target",
            "error_file": "None",
            "start_file": "eple",
            "stdout": "aggregator.stdout",
            "stderr": "aggregator.stderr",
            "stdin": "illgiveyousome",
            "argList": ["-o"],
            "environment": {"STATE": "awesome"},
            "max_running_minutes": 1,
        },
        {
            "name": "PI",
            "executable": "pi.py",
            "target_file": "my_target_file",
            "error_file": "error_file",
            "start_file": "some_start_file",
            "stdout": "pi.stdout",
            "stderr": "pi.stderr",
            "stdin": "input4thewin",
            "argList": ["-p", "8"],
            "environment": {"LOCATION": "earth"},
            "max_running_minutes": 12,
        },
        {
            "name": "OPTIMUS",
            "executable": "optimus.py",
            "target_file": "target",
            "error_file": "None",
            "start_file": "eple",
            "stdout": "optimus.stdout",
            "stderr": "optimus.stderr",
            "stdin": "illgiveyousome",
            "argList": ["-help"],
            "environment": {"PATH": "/ubertools/4.1"},
            "max_running_minutes": 1,
        },
    ]
    for step in result:
        step["environment"].update(ForwardModelStep.default_env)
    return result


forward_model_keywords = [
    "STDIN",
    "STDOUT",
    "STDERR",
    "EXECUTABLE",
    "TARGET_FILE",
    "ERROR_FILE",
    "START_FILE",
    "ARGLIST",
    "ENV",
    "MAX_RUNNING_MINUTES",
]

json_keywords = [
    "name",
    "executable",
    "target_file",
    "error_file",
    "start_file",
    "stdout",
    "stderr",
    "stdin",
    "max_running_minutes",
    "argList",
    "environment",
]


def str_none_sensitive(x):
    return str(x) if x is not None else None


DEFAULT_NAME = "default_step_name"


def _generate_step(
    name,
    executable,
    target_file,
    error_file,
    start_file,
    stdout,
    stderr,
    stdin,
    environment,
    arglist,
    max_running_minutes,
):
    config_file = DEFAULT_NAME if name is None else name

    values = [
        stdin,
        stdout,
        stderr,
        executable,
        target_file,
        error_file,
        start_file,
        None if arglist is None else " ".join(arglist),
        environment,
        str_none_sensitive(max_running_minutes),
    ]

    config_contents = ""
    for key, val in zip(forward_model_keywords, values, strict=False):
        if key == "ENV" and val:
            for k, v in val.items():
                config_contents += f"{key} {k} {v}\n"
        elif val is not None:
            config_contents += f"{key} {val}\n"

    with open(executable, "w", encoding="utf-8"):
        pass
    mode = os.stat(executable).st_mode
    mode |= stat.S_IXUSR | stat.S_IXGRP
    os.chmod(executable, stat.S_IMODE(mode))

    return forward_model_step_from_config_contents(config_contents, config_file, name)


def empty_list_if_none(list_):
    return [] if list_ is None else list_


def default_name_if_none(name):
    return DEFAULT_NAME if name is None else name


def create_std_file(config, std="stdout", step_index=None):
    if step_index is None:
        if config[std]:
            return f"{config[std]}"
        else:
            return f"{config['name']}.{std}"
    elif config[std]:
        return f"{config[std]}.{step_index}"
    else:
        return f"{config['name']}.{std}.{step_index}"


def validate_forward_model(forward_model, forward_model_config):
    assert forward_model.name == default_name_if_none(forward_model_config["name"])
    assert forward_model.executable == forward_model_config["executable"]
    assert forward_model.target_file == forward_model_config["target_file"]
    assert forward_model.error_file == forward_model_config["error_file"]
    assert forward_model.start_file == forward_model_config["start_file"]
    assert forward_model.stdout_file == create_std_file(
        forward_model_config, std="stdout"
    )
    assert forward_model.stderr_file == create_std_file(
        forward_model_config, std="stderr"
    )
    assert forward_model.stdin_file == forward_model_config["stdin"]
    assert (
        forward_model.max_running_minutes == forward_model_config["max_running_minutes"]
    )
    assert forward_model.arglist == empty_list_if_none(forward_model_config["argList"])
    if forward_model_config["environment"] is None:
        assert forward_model.environment == ForwardModelStep.default_env
    else:
        assert (
            forward_model.environment.keys()
            == forward_model_config["environment"].keys()
        )
        for key in forward_model_config["environment"]:
            assert (
                forward_model.environment[key]
                == forward_model_config["environment"][key]
            )


def generate_step_from_dict(forward_model_config):
    forward_model_config = copy.deepcopy(forward_model_config)
    forward_model_config["executable"] = os.path.join(
        os.getcwd(), forward_model_config["executable"]
    )
    forward_model = _generate_step(
        forward_model_config["name"],
        forward_model_config["executable"],
        forward_model_config["target_file"],
        forward_model_config["error_file"],
        forward_model_config["start_file"],
        forward_model_config["stdout"],
        forward_model_config["stderr"],
        forward_model_config["stdin"],
        forward_model_config["environment"],
        forward_model_config["argList"],
        forward_model_config["max_running_minutes"],
    )

    validate_forward_model(forward_model, forward_model_config)
    return forward_model


def set_up_forward_model(fm_steplist) -> list[ForwardModelStep]:
    return [generate_step_from_dict(step) for step in fm_steplist]


def verify_json_dump(fm_steplist, config, selected_steps, run_id):
    expected_default_env = {
        "_ERT_ITERATION_NUMBER": "0",
        "_ERT_REALIZATION_NUMBER": "0",
        "_ERT_RUNPATH": "./",
    }
    assert "config_path" in config
    assert "config_file" in config
    assert run_id == config["run_id"]
    assert len(selected_steps) == len(config["jobList"])

    for step_index, selected_step in enumerate(selected_steps):
        step = fm_steplist[selected_step]
        loaded_step = config["jobList"][step_index]

        # Since no argList is loaded as an empty list by forward_model
        arg_list_back_up = step["argList"]
        step["argList"] = empty_list_if_none(step["argList"])

        # Since name is set to default if none provided by forward_model
        name_back_up = step["name"]
        step["name"] = default_name_if_none(step["name"])

        for key in json_keywords:
            if key in {"stdout", "stderr"}:
                assert (
                    create_std_file(step, std=key, step_index=step_index)
                    == loaded_step[key]
                )
            elif key == "executable":
                assert step[key] in loaded_step[key]
            elif key == "environment" and step[key] is None:
                assert loaded_step[key] == expected_default_env
            elif key == "environment" and step[key] is not None:
                for k in step[key]:
                    if k not in ForwardModelStep.default_env:
                        assert step[key][k] == loaded_step[key][k]
                    else:
                        assert step[key][k] == ForwardModelStep.default_env[k]
                        assert loaded_step[key][k] == expected_default_env[k]
            else:
                assert step[key] == loaded_step[key]

        step["argList"] = arg_list_back_up
        step["name"] = name_back_up


@pytest.mark.usefixtures("use_tmpdir")
def test_config_path_and_file(context):
    run_id = "test_config_path_and_file_in_jobs_json"

    ert_config = ErtConfig(
        forward_model_steps=set_up_forward_model([]),
        substitutions=context,
        user_config_file="path_to_config_file/config.ert",
    )
    steps_json = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id=run_id,
    )
    assert "config_path" in steps_json
    assert "config_file" in steps_json
    assert steps_json["config_path"] == "path_to_config_file"
    assert steps_json["config_file"] == "config.ert"


@pytest.mark.usefixtures("use_tmpdir")
def test_no_steps(context):
    run_id = "test_no_jobs_id"

    ert_config = ErtConfig(
        forward_model_steps=set_up_forward_model([]),
        substitutions=context,
        user_config_file="path_to_config_file/config.ert",
    )

    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id=run_id,
    )

    verify_json_dump([], data, [], run_id)


@pytest.mark.usefixtures("use_tmpdir")
def test_one_step(fm_step_list, context):
    for i, step in enumerate(fm_step_list):
        run_id = "test_one_job"

        ert_config = ErtConfig(
            forward_model_steps=set_up_forward_model([step]),
            substitutions=context,
        )

        data = create_forward_model_json(
            context=ert_config.substitutions,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            user_config_file=ert_config.user_config_file,
            run_id=run_id,
        )

        verify_json_dump(fm_step_list, data, [i], run_id)


def run_all(fm_steplist, context):
    run_id = "run_all"
    ert_config = ErtConfig(
        forward_model_steps=set_up_forward_model(fm_steplist),
        substitutions=context,
    )

    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id=run_id,
    )

    verify_json_dump(fm_steplist, data, range(len(fm_steplist)), run_id)


@pytest.mark.usefixtures("use_tmpdir")
def test_all_steps(fm_step_list, context):
    run_all(fm_step_list, context)


@pytest.mark.usefixtures("use_tmpdir")
def test_various_null_fields(fm_step_list, context):
    for key in [
        "target_file",
        "error_file",
        "start_file",
        "stdout",
        "stderr",
        "max_running_minutes",
        "argList",
        "environment",
        "stdin",
    ]:
        fm_step_list[0][key] = None
        run_all(fm_step_list, context)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_env_vars_with_surrounded_by_brackets_are_ommitted_from_Jobs_json(
    caplog, fm_step_list, context
):
    forward_model_list: list[ForwardModelStep] = set_up_forward_model(fm_step_list)
    forward_model_list[0].environment["ENV_VAR"] = "<SOME_BRACKETS>"
    run_id = "test_no_jobs_id"

    ert_config = ErtConfig(
        forward_model_steps=forward_model_list, substitutions=context
    )

    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id=run_id,
    )

    assert "Environment variable ENV_VAR skipped due to" in caplog.text
    assert "ENV_VAR" not in data["jobList"][0]["environment"]


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("job", "forward_model", "expected_args"),
    [
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST WORD_A
            """
            ),
            "FORWARD_MODEL job_name",
            ["WORD_A"],
            id="No argument",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ARGUMENTA>
            """
            ),
            dedent(
                """
            DEFINE <ARGUMENTA> foo
            FORWARD_MODEL job_name
            """
            ),
            ["foo"],
            id="Global arguments are used inside job definition",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ARGUMENT>
            """
            ),
            "FORWARD_MODEL job_name(<ARGUMENT>=yy)",
            ["yy"],
            id="With argument",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
                """
            ),
            "DEFINE <TO_BE_DEFINED> <ARGUMENTB>\n"
            "FORWARD_MODEL job_name(<ARGUMENTA>=configured_argumentA,"
            " <TO_BE_DEFINED>=configured_argumentB)",
            [
                "configured_argumentA",
                "<ARGUMENTB>",
                "<ARGUMENTC>",
            ],
            id="Keywords in argument list are not substituted, "
            "so argument B gets no value",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
                """
            ),
            dedent(
                """
            DEFINE <ARGUMENTB> DEFINED_ARGUMENTB_VALUE
            FORWARD_MODEL job_name(<ARGUMENTB>=configured_argumentB)
            """
            ),
            ["DEFAULT_ARGA_VALUE", "configured_argumentB", "<ARGUMENTC>"],
            id="Resolved argument given by argument list, not overridden by global",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
            """
            ),
            "FORWARD_MODEL job_name()",
            ["DEFAULT_ARGA_VALUE", "<ARGUMENTB>", "<ARGUMENTC>"],
            id="No args, parenthesis, gives default argument A",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            DEFAULT <ARGUMENTA> DEFAULT_ARGA_VALUE
            ARGLIST <ARGUMENTA> <ARGUMENTB> <ARGUMENTC>
            """
            ),
            "FORWARD_MODEL job_name",
            ["DEFAULT_ARGA_VALUE", "<ARGUMENTB>", "<ARGUMENTC>"],
            id="No args, gives default argument A",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ITER>
            """
            ),
            "FORWARD_MODEL job_name(<ITER>=<ITER>)",
            ["0"],
            id="This style of args works without infinite substitution loop.",
        ),
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ECLBASE>
            """
            ),
            "FORWARD_MODEL job_name(<ECLBASE>=A/<ECLBASE>)",
            ["A/ECLBASE0"],
            id="The NOSIM job takes <ECLBASE> as args. Expect no infinite loop.",
        ),
    ],
)
def test_forward_model_job(job, forward_model, expected_args):
    Path("job_file").write_text(job, encoding="utf-8")

    # Write a minimal config file
    Path("config_file.ert").write_text(
        "NUM_REALIZATIONS 1\nINSTALL_JOB job_name job_file\n" + forward_model,
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file("config_file.ert")

    forward_model = ert_config.forward_model_steps

    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )
    assert len(forward_model) == 1
    assert data["jobList"][0]["argList"] == expected_args


@pytest.mark.usefixtures("use_tmpdir")
def test_that_config_path_is_the_directory_of_the_main_ert_config():
    os.mkdir("jobdir")
    Path("jobdir/job_file").write_text(
        dedent(
            """
            EXECUTABLE echo
            ARGLIST <CONFIG_PATH>
            """
        ),
        encoding="utf-8",
    )

    # Write a minimal config file
    Path("config_file.ert").write_text(
        "NUM_REALIZATIONS 1\n"
        "INSTALL_JOB job_name jobdir/job_file\n"
        "FORWARD_MODEL job_name",
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file("config_file.ert")
    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )
    assert data["jobList"][0]["argList"] == [os.getcwd()]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_private_over_global_args_gives_logging_message(caplog):
    caplog.set_level(logging.INFO)
    Path("job_file").write_text(
        dedent(
            """
            EXECUTABLE echo
            ARGLIST <ARG>
            ARG_TYPE 0 STRING
            """
        ),
        encoding="utf-8",
    )

    # Write a minimal config file
    Path("config_file.ert").write_text(
        "NUM_REALIZATIONS 1\n"
        "DEFINE <ARG> A\n"
        "INSTALL_JOB job_name job_file\n"
        "FORWARD_MODEL job_name(<ARG>=B)",
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file("config_file.ert")
    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )

    fm_data = data["jobList"][0]

    assert len(ert_config.forward_model_steps) == 1
    assert fm_data["argList"] == ["B"]
    assert "Private arg '<ARG>':'B' chosen over global 'A'" in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
def test_that_private_over_global_args_does_not_give_logging_message_for_argpassing(
    caplog,
):
    caplog.set_level(logging.INFO)
    Path("job_file").write_text(
        dedent(
            """
            EXECUTABLE echo
            ARGLIST <ARG>
            ARG_TYPE 0 STRING
            """
        ),
        encoding="utf-8",
    )

    # Write a minimal config file
    Path("config_file.ert").write_text(
        "NUM_REALIZATIONS 1\n"
        "DEFINE <ARG> A\n"
        "INSTALL_JOB job_name job_file\n"
        "FORWARD_MODEL job_name(<ARG>=<ARG>)",
        encoding="utf-8",
    )

    ert_config = ErtConfig.from_file("config_file.ert")
    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )

    fm_data = data["jobList"][0]
    assert len(ert_config.forward_model_steps) == 1
    assert fm_data["argList"] == ["A"]
    assert "Private arg '<ARG>':'<ARG>' chosen over global 'A'" not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("job", "forward_model", "expected_args"),
    [
        pytest.param(
            dedent(
                """
            EXECUTABLE echo
            ARGLIST <ARG>
            """
            ),
            "DEFINE <ENV> $ENV\nFORWARD_MODEL job_name(<ARG>=<ENV>)",
            ["env_value"],
            id=(
                "Test that the environment variable $ENV "
                "is put into the forward model step"
            ),
        ),
    ],
)
def test_that_environment_variables_are_set_in_forward_model(
    monkeypatch, job, forward_model, expected_args
):
    monkeypatch.setenv("ENV", "env_value")
    Path("job_file").write_text(job, encoding="utf-8")

    with open("config_file.ert", "w", encoding="utf-8") as fout:
        # Write a minimal config file
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        """
            )
        )
        fout.write("INSTALL_JOB job_name job_file\n")
        fout.write(forward_model)

    ert_config = ErtConfig.from_file("config_file.ert")
    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )

    forward_model_list = ert_config.forward_model_steps
    assert len(forward_model_list) == 1
    assert data["jobList"][0]["argList"] == expected_args


def test_that_executables_in_path_are_not_made_realpath(tmp_path):
    """
    Before 9e4fb6aed0d2650f90fa59a24ba2e7e7cac19a0c executables in path would
    be resolved with `which` and made into an abspath to that executable. When
    running the forward model on a different machine, that abspath may no
    longer be valid. Also, if the user wants to give an abspath, that is still
    possible.

    Therefore, the behavior was changed to what is being tested for here.
    """
    (tmp_path / "echo_job").write_text("EXECUTABLE echo\n ARGLIST <MSG>")

    config_file = tmp_path / "config.ert"
    config_file.write_text(
        "NUM_REALIZATIONS 1\n"
        "INSTALL_JOB echo echo_job\n"
        'FORWARD_MODEL echo(<MSG>="hello")\n'
    )

    ert_config = ErtConfig.from_file(str(config_file))
    data = create_forward_model_json(
        context=ert_config.substitutions,
        forward_model_steps=ert_config.forward_model_steps,
        env_vars=ert_config.env_vars,
        user_config_file=ert_config.user_config_file,
        run_id="",
    )

    assert data["jobList"][0]["executable"] == "echo"
