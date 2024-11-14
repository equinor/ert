import logging

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
    assert "flow" in project_code and "rms" in project_code


@pytest.mark.parametrize("queue_system", ["LSF", "TORQUE", "SLURM"])
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


@pytest.mark.parametrize("invalid_queue_system", ["VOID", "BLABLA", "GENERIC", "*"])
def test_that_an_invalid_queue_system_provided_raises_validation_error(
    invalid_queue_system,
):
    """There is actually a "queue-system" called GENERIC, but it is
    only there for making queue options global, it should not be
    possible to try to use it"""
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=f"'QUEUE_SYSTEM' argument 1 must be one of .* was '{invalid_queue_system}'",
    ):
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {invalid_queue_system}\n"
        )


@pytest.mark.parametrize(
    "queue_system, invalid_option", [("LOCAL", "BSUB_CMD"), ("TORQUE", "BOGUS")]
)
def test_that_invalid_queue_option_raises_validation_error(
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
        ).queue_config.realization_memory
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
    ],
)
def test_realization_memory_unit_support(memory_spec: str, expected_bytes: int):
    assert (
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nREALIZATION_MEMORY {memory_spec}\n"
        ).queue_config.realization_memory
        == expected_bytes
    )


@pytest.mark.parametrize("invalid_memory_spec", ["-1", "-1b", "b", "4ub"])
def test_invalid_realization_memory(invalid_memory_spec: str):
    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 1\nREALIZATION_MEMORY {invalid_memory_spec}\n"
        )


def test_conflicting_realization_slurm_memory():
    with (
        pytest.raises(ConfigValidationError),
        pytest.warns(ConfigWarning, match="deprecated"),
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "REALIZATION_MEMORY 10Mb\n"
            "QUEUE_SYSTEM SLURM\n"
            "QUEUE_OPTION SLURM MEMORY 20M\n"
        )


def test_conflicting_realization_slurm_memory_per_cpu():
    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "REALIZATION_MEMORY 10Mb\n"
            "QUEUE_SYSTEM SLURM\n"
            "QUEUE_OPTION SLURM MEMORY_PER_CPU 20M\n"
        )


def test_conflicting_realization_openpbs_memory_per_job():
    with (
        pytest.raises(ConfigValidationError),
        pytest.warns(ConfigWarning, match="deprecated"),
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "REALIZATION_MEMORY 10Mb\n"
            "QUEUE_SYSTEM TORQUE\n"
            "QUEUE_OPTION TORQUE MEMORY_PER_JOB 20mb\n"
        )


def test_conflicting_realization_openpbs_memory_per_job_but_slurm_activated_only_warns():
    with pytest.warns(ConfigWarning):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "REALIZATION_MEMORY 10Mb\n"
            "QUEUE_SYSTEM SLURM\n"
            "QUEUE_OPTION TORQUE MEMORY_PER_JOB 20mb\n"
        )


@pytest.mark.parametrize("torque_memory_with_unit_str", ["gb", "mb", "1 gb"])
def test_that_invalid_memory_pr_job_raises_validation_error(
    torque_memory_with_unit_str,
):
    with (
        pytest.raises(ConfigValidationError),
        pytest.warns(ConfigWarning, match="deprecated"),
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM TORQUE\n"
            f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {torque_memory_with_unit_str}"
        )


@pytest.mark.parametrize(
    "queue_system, queue_system_option",
    [("LSF", "LSF_QUEUE"), ("SLURM", "SQUEUE"), ("TORQUE", "QUEUE")],
)
def test_that_overwriting_QUEUE_OPTIONS_warns(
    queue_system, queue_system_option, caplog
):
    with caplog.at_level(logging.INFO):
        ErtConfig.from_file_contents(
            user_config_contents="NUM_REALIZATIONS 1\n"
            f"QUEUE_SYSTEM {queue_system}\n"
            f"QUEUE_OPTION {queue_system} {queue_system_option} test_1\n"
            f"QUEUE_OPTION {queue_system} MAX_RUNNING 10\n",
            site_config_contents="JOB_SCRIPT job_dispatch.py\n"
            f"QUEUE_SYSTEM {queue_system}\n"
            f"QUEUE_OPTION {queue_system} {queue_system_option} test_0\n"
            f"QUEUE_OPTION {queue_system} MAX_RUNNING 10\n",
        )
    assert (
        f"Overwriting QUEUE_OPTION {queue_system} {queue_system_option}: \n Old value:"
        " test_0 \n New value: test_1"
    ) in caplog.text and (
        f"Overwriting QUEUE_OPTION {queue_system} MAX_RUNNING: \n Old value:"
        " 10 \n New value: 10"
    ) not in caplog.text


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
        ("TORQUE", "NUM_NODES", "3.5", "should be a valid integer"),
    ],
)
def test_wrong_config_option_types(queue_system, queue_option, queue_value, err_msg):
    file_contents = (
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {queue_option} {queue_value}\n"
    )

    with pytest.raises(ConfigValidationError, match=err_msg):
        if queue_system == "TORQUE":
            with pytest.warns(ConfigWarning, match="deprecated"):
                ErtConfig.from_file_contents(file_contents)
        else:
            ErtConfig.from_file_contents(file_contents)


def test_that_configuring_another_queue_system_gives_warning():
    with pytest.warns(ConfigWarning, match="should be a valid number"):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM LSF\n"
            "QUEUE_OPTION SLURM SQUEUE_TIMEOUT ert\n"
        )


def test_that_slurm_queue_mem_options_are_validated():
    with pytest.raises(ConfigValidationError) as e:
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM SLURM\n"
            "QUEUE_OPTION SLURM MEMORY_PER_CPU 5mb\n"
        )

    info = e.value.errors[0]

    assert "Value error, wrong memory format. Got input '5mb'." in info.message
    assert info.line == 3
    assert info.column == 35
    assert info.end_column == info.column + 3


@pytest.mark.parametrize(
    "mem_per_job",
    ["5gb", "5mb", "5kb"],
)
def test_that_valid_torque_queue_mem_options_are_ok(mem_per_job):
    with pytest.warns(ConfigWarning, match="deprecated"):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM SLURM\n"
            f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {mem_per_job}\n"
        )


@pytest.mark.parametrize(
    "mem_per_job",
    ["5", "5g"],
)
def test_that_torque_queue_mem_options_are_corrected(mem_per_job: str):
    with (
        pytest.raises(ConfigValidationError) as e,
        pytest.warns(ConfigWarning, match="deprecated"),
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            "QUEUE_SYSTEM TORQUE\n"
            f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {mem_per_job}\n"
        )

    info = e.value.errors[0]

    assert (
        f"Value error, wrong memory format. Got input '{mem_per_job}'." in info.message
    )
    assert info.line == 3
    assert info.column == 36
    assert info.end_column == info.column + len(mem_per_job)


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


@pytest.mark.parametrize("queue_system", ["LSF", "GENERIC"])
def test_multiple_submit_sleep_keywords(queue_system):
    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\n"
        "QUEUE_SYSTEM LSF\n"
        "QUEUE_OPTION LSF SUBMIT_SLEEP 10\n"
        f"QUEUE_OPTION {queue_system} SUBMIT_SLEEP 42\n"
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
            "NUM_REALIZATIONS 1\n" f"MAX_SUBMIT {max_submit_value}\n"
        )


@pytest.mark.parametrize(
    "queue_system, key, value",
    [
        ("LSF", "MAX_RUNNING", 50),
        ("SLURM", "MAX_RUNNING", 50),
        ("TORQUE", "MAX_RUNNING", 50),
        ("LSF", "SUBMIT_SLEEP", 4.2),
        ("SLURM", "SUBMIT_SLEEP", 4.2),
        ("TORQUE", "SUBMIT_SLEEP", 4.2),
    ],
)
def test_global_queue_options(queue_system, key, value):
    def _check_results(contents):
        ert_config = ErtConfig.from_file_contents(contents)
        if key == "MAX_RUNNING":
            assert ert_config.queue_config.max_running == value
        elif key == "SUBMIT_SLEEP":
            assert ert_config.queue_config.submit_sleep == value
        else:
            raise KeyError("Unexpected key")

    _check_results(
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {key} 10\n"
        f"QUEUE_OPTION GENERIC {key} {value}\n"
    )

    _check_results(f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {queue_system}\n{key} {value}\n")


@pytest.mark.parametrize(
    "queue_system, key, value",
    [
        ("LSF", "MAX_RUNNING", 50),
        ("SLURM", "MAX_RUNNING", 50),
        ("TORQUE", "MAX_RUNNING", 50),
        ("LSF", "SUBMIT_SLEEP", 4.2),
        ("SLURM", "SUBMIT_SLEEP", 4.2),
        ("TORQUE", "SUBMIT_SLEEP", 4.2),
    ],
)
def test_global_config_key_does_not_overwrite_queue_options(queue_system, key, value):
    def _check_results(contents):
        ert_config = ErtConfig.from_file_contents(contents)
        if key == "MAX_RUNNING":
            assert ert_config.queue_config.max_running == value
        elif key == "SUBMIT_SLEEP":
            assert ert_config.queue_config.submit_sleep == value
        else:
            raise KeyError("Unexpected key")

    _check_results(
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {key} {value}\n"
        f"{key} {value + 42}\n"
    )

    _check_results(
        "NUM_REALIZATIONS 1\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION GENERIC {key} {value}\n"
        f"{key} {value + 42}\n"
    )


@pytest.mark.parametrize(
    "queue_system, key, value",
    [
        ("LSF", "MAX_RUNNING", -50),
        ("SLURM", "MAX_RUNNING", -50),
        ("TORQUE", "MAX_RUNNING", -50),
        ("LSF", "SUBMIT_SLEEP", -4.2),
        ("SLURM", "SUBMIT_SLEEP", -4.2),
        ("TORQUE", "SUBMIT_SLEEP", -4.2),
    ],
)
def test_wrong_generic_queue_option_raises_validation_error(queue_system, key, value):
    with pytest.raises(
        ConfigValidationError, match="Input should be greater than or equal to 0"
    ):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n"
            f"QUEUE_SYSTEM {queue_system}\n"
            f"QUEUE_OPTION GENERIC {key} {value}\n"
        )

    error_msg = (
        "must have a positive integer value"
        if key == "MAX_RUNNING"
        else "Input should be greater than or equal to 0"
    )
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file_contents(
            "NUM_REALIZATIONS 1\n" f"QUEUE_SYSTEM {queue_system}\n" f"{key} {value}\n"
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
