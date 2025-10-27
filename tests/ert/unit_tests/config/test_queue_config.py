import logging
import shutil

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
    QueueConfig,
    QueueSystem,
)
from ert.config.parsing import ConfigKeys
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.plugins import ErtRuntimePlugins
from ert.scheduler import LocalDriver, LsfDriver, OpenPBSDriver, SlurmDriver


def test_create_local_copy_is_a_copy_with_local_queue_system():
    queue_config = QueueConfig(queue_system=QueueSystem.LSF)
    assert queue_config.queue_system == QueueSystem.LSF
    local_queue_config = queue_config.create_local_copy()
    assert local_queue_config.queue_system == QueueSystem.LOCAL
    assert isinstance(local_queue_config.queue_options, LocalQueueOptions)


@pytest.mark.parametrize("value", [True, False])
def test_stop_long_running_is_set_from_corresponding_keyword(value):
    assert (
        QueueConfig.from_dict({ConfigKeys.STOP_LONG_RUNNING: value}).stop_long_running
        == value
    )
    assert QueueConfig(stop_long_running=value).stop_long_running == value


@pytest.mark.parametrize("queue_system", ["LSF", "TORQUE", "SLURM"])
def test_project_code_is_set_when_forward_model_contains_selected_simulator(
    queue_system,
):
    queue_config = QueueConfig.from_dict(
        {
            ConfigKeys.FORWARD_MODEL: [("FLOW",), ("RMS",)],
            ConfigKeys.QUEUE_SYSTEM: queue_system,
        }
    )
    project_code = queue_config.queue_options.project_code

    assert project_code is not None
    assert "flow" in project_code
    assert "rms" in project_code


@pytest.mark.parametrize(
    "queue_system", [QueueSystem.LSF, QueueSystem.TORQUE, QueueSystem.SLURM]
)
def test_project_code_is_not_overwritten_if_set_in_config(queue_system):
    queue_config = QueueConfig.from_dict(
        {
            ConfigKeys.FORWARD_MODEL: [("FLOW",), ("RMS",)],
            ConfigKeys.QUEUE_SYSTEM: queue_system,
            "QUEUE_OPTION": [
                [queue_system, "PROJECT_CODE", "test_code"],
            ],
        }
    )
    assert queue_config.queue_options.project_code == "test_code"


@pytest.mark.parametrize("invalid_queue_system", ["VOID", "BLABLA", "*"])
def test_that_the_first_argument_to_queue_option_must_be_a_known_queue_system(
    invalid_queue_system,
):
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=(
            f"'QUEUE_SYSTEM' argument 1 must be one of .* was '{invalid_queue_system}'"
        ),
    ):
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {invalid_queue_system}\n"
        )


@pytest.mark.parametrize(
    "queue_system, invalid_option",
    [(QueueSystem.LOCAL, "BSUB_CMD"), (QueueSystem.TORQUE, "BOGUS")],
)
def test_that_the_second_argument_to_queue_option_must_be_a_known_option_for_the_system(
    queue_system, invalid_option
):
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=f"Invalid QUEUE_OPTION for {queue_system}: '{invalid_option}'",
    ):
        _ = ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {queue_system}\n"
            f"QUEUE_OPTION {queue_system} {invalid_option}"
        )


@st.composite
def memory_with_unit(draw):
    memory_value = draw(st.integers(min_value=1, max_value=10000))
    unit = draw(
        st.sampled_from(["gb", "mb", "tb", "pb", "Kb", "Gb", "Mb", "Pb", "b", "B", ""])
    )
    return f"{memory_value}{unit}"


@given(memory_with_unit())
def test_supported_memory_units_to_realization_memory(
    memory_with_unit,
):
    assert (
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nREALIZATION_MEMORY {memory_with_unit}\n"
        ).queue_config.queue_options.realization_memory
        > 0
    )


@pytest.mark.parametrize(
    "memory_spec, expected_bytes",
    [
        ("1", 1),
        ("1b", 1),
        ("10b", 10),
        ("10kb", 10 * 1024),
        ("10mb", 10 * 1024**2),
        ("10gb", 10 * 1024**3),
        ("10Mb", 10 * 1024**2),
        ("10Gb", 10 * 1024**3),
        ("10Tb", 10 * 1024**4),
        ("10Pb", 10 * 1024**5),
        ("'  10      Gb  '", 10 * 1024**3),
        ("'10   GB'", 10 * 1024**3),
        (0, 0),
    ],
)
def test_realization_memory_unit_support(memory_spec: str, expected_bytes: int):
    assert (
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nREALIZATION_MEMORY {memory_spec}\n"
        ).queue_config.queue_options.realization_memory
        == expected_bytes
    )


@pytest.mark.parametrize(
    "invalid_memory_spec, error_message",
    [
        ("-1", "Negative memory does not make sense"),
        ("'      -2'", "Negative memory does not make sense"),
        ("-1b", "Negative memory does not make sense in -1b"),
        ("b", "Invalid memory string"),
        ("'kljh3 k34f15gg.  asd '", "Invalid memory string"),
        ("'kljh3 1gb'", "Invalid memory string"),
        ("' 2gb 3k 1gb'", "Invalid memory string"),
        ("4ub", "Unknown memory unit"),
    ],
)
def test_invalid_realization_memory(invalid_memory_spec: str, error_message: str):
    with pytest.raises(
        ConfigValidationError, match=rf"Line 2 \(Column \d+-\d+\): {error_message}"
    ):
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nREALIZATION_MEMORY {invalid_memory_spec}\n"
        )


@pytest.fixture(
    params=[
        (QueueSystem.LSF, LsfQueueOptions, "LSF_QUEUE"),
        (QueueSystem.SLURM, SlurmQueueOptions, "SQUEUE"),
        (QueueSystem.TORQUE, TorqueQueueOptions, "QUEUE"),
    ]
)
def queue_that_overrides_site_config(request, caplog):
    queue_system, queue_options_cls, queue_system_option = request.param

    config_dict = ErtConfig._config_dict_from_contents(
        f"""
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM {queue_system}
        QUEUE_OPTION {queue_system} {queue_system_option} test_1
        QUEUE_OPTION {queue_system} MAX_RUNNING 10
        QUEUE_OPTION {queue_system} NUM_CPU 9
        QUEUE_OPTION {queue_system} SUBMIT_SLEEP 1337
        QUEUE_OPTION {queue_system} JOB_SCRIPT usr_dispatch.py
        QUEUE_OPTION {queue_system} JOB_SCRIPT usr_dispatch2.py
        QUEUE_OPTION {queue_system} JOB_SCRIPT usr_dispatch3.py
        """,
        "config.ert",
    )

    with caplog.at_level(logging.INFO):
        queue_config = QueueConfig.from_dict(
            config_dict,
            site_queue_options=queue_options_cls(
                **{
                    "name": queue_system,
                    queue_system_option.lower(): "the_site_queue",
                    "max_running": 2,
                    "num_cpu": 3,
                    "submit_sleep": 4,
                    "job_script": "site_job_script.sh",
                }
            ),
        )

    # return both params and logs so tests can use them
    return queue_config, queue_system, queue_system_option, caplog.text


def test_that_overwriting_QUEUE_OPTIONS_warns(queue_that_overrides_site_config):
    _, queue_system, queue_system_option, captured_log = (
        queue_that_overrides_site_config
    )

    assert (
        f"Overwriting site config setting: "
        f"{queue_system_option.lower()}=the_site_queue with "
        f"QUEUE_OPTION {queue_system.upper()} {queue_system_option} test_1"
        in captured_log
    )

    assert (
        f"Overwriting site config setting: max_running=2 with "
        f"QUEUE_OPTION {queue_system.upper()} MAX_RUNNING 10" in captured_log
    )

    assert (
        f"Overwriting site config setting: num_cpu=3 with "
        f"QUEUE_OPTION {queue_system.upper()} NUM_CPU 9" in captured_log
    )

    assert (
        f"Overwriting site config setting: submit_sleep=4.0 with "
        f"QUEUE_OPTION {queue_system.upper()} SUBMIT_SLEEP 1337" in captured_log
    )

    assert (
        f"Overwriting site config setting: job_script=site_job_script.sh with "
        f"QUEUE_OPTION {queue_system.upper()} JOB_SCRIPT usr_dispatch.py"
        in captured_log
    )

    assert (
        f"Overwriting QUEUE_OPTION {queue_system.upper()} JOB_SCRIPT: \n "
        f"Old value: usr_dispatch.py \n New value: usr_dispatch2.py\n" in captured_log
    )

    assert (
        f"Overwriting QUEUE_OPTION {queue_system.upper()} JOB_SCRIPT: \n "
        f"Old value: usr_dispatch2.py \n New value: usr_dispatch3.py\n" in captured_log
    )


def test_that_user_given_queue_settings_overwrites_site_config(
    queue_that_overrides_site_config,
):
    queue_config, _, queue_system_option, _ = queue_that_overrides_site_config

    actual = queue_config.queue_options.model_dump(exclude_unset=True)
    expected = {
        "max_running": 10,
        "submit_sleep": 1337.0,
        "job_script": "usr_dispatch3.py",
        "num_cpu": 9,
        queue_system_option.lower(): "test_1",
    }
    assert {k: actual[k] for k in expected} == expected


@pytest.mark.parametrize(
    "queue_system, queue_system_option",
    [("LSF", "LSF_QUEUE"), ("SLURM", "SQUEUE")],
)
def test_initializing_empty_config_queue_options_resets_to_default_value(
    queue_system, queue_system_option
):
    config_object = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {queue_system_option}\n"
        f"QUEUE_OPTION {queue_system} MAX_RUNNING\n"
    )

    if queue_system == "LSF":
        assert config_object.queue_config.queue_options.lsf_queue is None
    if queue_system == "SLURM":
        assert config_object.queue_config.queue_options.squeue == "squeue"
    assert config_object.queue_config.queue_options.max_running == 0


@pytest.mark.parametrize(
    "queue_system, queue_option, queue_value, err_msg",
    [
        ("SLURM", "SQUEUE_TIMEOUT", "5a", "should be a valid number"),
    ],
)
def test_wrong_config_option_types(queue_system, queue_option, queue_value, err_msg):
    file_contents = (
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {queue_option} {queue_value}\n"
    )

    with pytest.raises(ConfigValidationError, match=err_msg):
        ErtConfig.from_file_contents(file_contents)


def test_that_configuring_another_queue_system_gives_warning():
    with pytest.warns(ConfigWarning, match="should be a valid number"):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM LSF\n"
            "QUEUE_OPTION SLURM SQUEUE_TIMEOUT ert\n"
        )


def test_max_running_property():
    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        "QUEUE_SYSTEM TORQUE\n"
        "QUEUE_OPTION TORQUE MAX_RUNNING 17\n"
        "QUEUE_OPTION TORQUE MAX_RUNNING 19\n"
        "QUEUE_OPTION LOCAL MAX_RUNNING 11\n"
        "QUEUE_OPTION LOCAL MAX_RUNNING 13\n"
    )

    assert config.queue_config.queue_system == QueueSystem.TORQUE
    assert config.queue_config.max_running == 19


def test_multiple_submit_sleep_keywords():
    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        "QUEUE_SYSTEM LSF\n"
        "QUEUE_OPTION LSF SUBMIT_SLEEP 10\n"
        "QUEUE_OPTION LSF SUBMIT_SLEEP 42\n"
        "QUEUE_OPTION TORQUE SUBMIT_SLEEP 22\n"
    )
    assert config.queue_config.submit_sleep == 42


def test_multiple_max_submit_keywords():
    assert (
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\nMAX_SUBMIT 10\nMAX_SUBMIT 42\n"
        ).queue_config.max_submit
        == 42
    )


@pytest.mark.parametrize(
    "max_submit_value, error_msg",
    [
        (-1, "must have a positive integer value as argument"),
        (0, "must have a positive integer value as argument"),
        (1.5, "must have an integer value as argument"),
    ],
)
def test_wrong_max_submit_raises_validation_error(max_submit_value, error_msg):
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nMAX_SUBMIT {max_submit_value}\n"
        )


@pytest.mark.parametrize(
    "key, value",
    [
        ("MAX_RUNNING", -50),
        ("SUBMIT_SLEEP", -4.2),
    ],
)
def test_invalid_queue_option_value_raises_validation_error(key, value):
    for queue_system in ["LSF", "SLURM", "TORQUE", "LOCAL"]:
        with pytest.raises(
            ConfigValidationError, match="Input should be greater than or equal to 0"
        ):
            ErtConfig.from_file_contents(
                "NUM_REALIZATIONS 1\n"
                f"QUEUE_SYSTEM {queue_system}\n"
                f"QUEUE_OPTION {queue_system} {key} {value}\n"
            )

        error_msg = (
            "must have a positive integer value"
            if key == "MAX_RUNNING"
            else "Input should be greater than or equal to 0"
        )
        with pytest.raises(ConfigValidationError, match=error_msg):
            ErtConfig.from_file_contents(
                f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {queue_system}\n{key} {value}\n"
            )


@pytest.mark.parametrize(
    "queue_system",
    (QueueSystem.LSF, QueueSystem.TORQUE, QueueSystem.LOCAL, QueueSystem.SLURM),
)
def test_driver_initialization_from_defaults(queue_system):
    if queue_system == QueueSystem.LSF:
        LsfDriver(**LsfQueueOptions().driver_options)
    if queue_system == QueueSystem.TORQUE:
        OpenPBSDriver(**TorqueQueueOptions().driver_options)
    if queue_system == QueueSystem.LOCAL:
        LocalDriver(**LocalQueueOptions().driver_options)
    if queue_system == QueueSystem.SLURM:
        SlurmDriver(**SlurmQueueOptions().driver_options)


@pytest.mark.parametrize(
    "venv, expected", [("my_env", "source my_env/bin/activate"), (None, "")]
)
def test_default_activate_script_generation(expected, monkeypatch, venv):
    if venv:
        monkeypatch.setenv("VIRTUAL_ENV", venv)
    else:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    options = LocalQueueOptions()
    assert options.activate_script == expected


@pytest.mark.parametrize(
    "env, expected",
    [
        ("my_env", 'eval "$(conda shell.bash hook)" && conda activate my_env'),
    ],
)
def test_conda_activate_script_generation(expected, monkeypatch, env):
    monkeypatch.setenv("VIRTUAL_ENV", "")
    monkeypatch.setenv("CONDA_ENV", env)
    options = LocalQueueOptions(name="local")
    assert options.activate_script == expected


@pytest.mark.parametrize(
    "env, expected",
    [("my_env", "source my_env/bin/activate")],
)
def test_multiple_activate_script_generation(expected, monkeypatch, env):
    monkeypatch.setenv("VIRTUAL_ENV", env)
    monkeypatch.setenv("CONDA_ENV", env)
    options = LocalQueueOptions(name="local")
    assert options.activate_script == expected


def test_default_max_runtime_is_unlimited():
    assert QueueConfig.from_dict({}).max_runtime is None
    assert QueueConfig().max_runtime is None


@given(st.integers(min_value=1))
def test_max_runtime_is_set_from_corresponding_keyword(value):
    assert QueueConfig.from_dict({ConfigKeys.MAX_RUNTIME: value}).max_runtime == value
    assert QueueConfig(max_runtime=value).max_runtime == value


def test_that_job_script_from_queue_options_takes_precedence_over_global(
    copy_poly_case,
):
    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        "JOB_SCRIPT poly_eval.py\n"
        "QUEUE_SYSTEM LSF\n"
        "QUEUE_OPTION LSF JOB_SCRIPT fm_dispatch_lsf.py\n"
    )
    assert config.queue_config.queue_options.job_script == "fm_dispatch_lsf.py"


def test_that_site_queue_options_are_ignored_with_differing_user_queue_system_arg():
    config = ErtConfig.with_plugins(
        ErtRuntimePlugins(queue_options=LsfQueueOptions())
    ).from_file_contents("NUM_REALIZATIONS 1\nQUEUE_SYSTEM LOCAL\n")

    assert config.queue_config.queue_system == QueueSystem.LOCAL


def test_that_job_script_precedence_is_user_then_site_then_defaults(caplog):
    config_dict = ErtConfig._config_dict_from_contents(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL
        QUEUE_OPTION LOCAL JOB_SCRIPT usr_dispatch3.py
        """,
        "config.ert",
    )

    queue_config = QueueConfig.from_dict(
        config_dict,
        site_queue_options=LocalQueueOptions(job_script="site_job_script.sh"),
    )

    assert queue_config.queue_options.job_script == "usr_dispatch3.py"


def test_that_usr_none_job_script_defaults_to_site(caplog):
    config_dict = ErtConfig._config_dict_from_contents(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL
        """,
        "config.ert",
    )

    queue_config = QueueConfig.from_dict(
        config_dict,
        site_queue_options=LocalQueueOptions(job_script="site_job_script.sh"),
    )

    assert queue_config.queue_options.job_script == "site_job_script.sh"


def test_that_usr_none_site_none_job_script_defaults_to_fmdispatch(caplog):
    config_dict = ErtConfig._config_dict_from_contents(
        """
        NUM_REALIZATIONS 1
        QUEUE_SYSTEM LOCAL
        """,
        "config.ert",
    )

    queue_config = QueueConfig.from_dict(
        config_dict,
        site_queue_options=LocalQueueOptions(),
    )

    expected_default_job_script = shutil.which("fm_dispatch.py") or "fm_dispatch.py"
    assert queue_config.queue_options.job_script == expected_default_job_script
