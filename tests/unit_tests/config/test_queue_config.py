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
    assert queue_config.create_local_copy().queue_system == QueueSystem.LOCAL


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


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_that_an_invalid_queue_system_provided_raises_validation_error(num_real):
    filename = "config.ert"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS {num_real}\nQUEUE_SYSTEM VOID\n")

    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="'QUEUE_SYSTEM' argument 1 must be one of .* was 'VOID'",
    ):
        _ = ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize(
    "queue_system, invalid_option", [("LOCAL", "BSUB_CMD"), ("TORQUE", "BOGUS")]
)
def test_that_invalid_queue_option_raises_validation_error(
    queue_system, invalid_option
):
    filename = "config.ert"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {invalid_option}")

    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=f"Invalid QUEUE_OPTION for {queue_system}: '{invalid_option}'",
    ):
        _ = ErtConfig.from_file(filename)


@st.composite
def memory_with_unit(draw):
    memory_value = draw(st.integers(min_value=1, max_value=10000))
    unit = draw(
        st.sampled_from(["gb", "mb", "tb", "pb", "Kb", "Gb", "Mb", "Pb", "b", "B", ""])
    )
    return f"{memory_value}{unit}"


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(memory_with_unit())
def test_supported_memory_units_to_realization_memory(
    memory_with_unit,
):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"REALIZATION_MEMORY {memory_with_unit}\n")

    assert ErtConfig.from_file(filename).queue_config.realization_memory > 0


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
def test_realization_memory_unit_support(memory_spec: str, expected_bytes, tmpdir):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"REALIZATION_MEMORY {memory_spec}\n")

    assert (
        ErtConfig.from_file(filename).queue_config.realization_memory == expected_bytes
    )


@pytest.mark.parametrize("invalid_memory_spec", ["-1", "-1b", "b", "4ub"])
def test_invalid_realization_memory(invalid_memory_spec: str, tmpdir):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"REALIZATION_MEMORY {invalid_memory_spec}\n")

    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file(filename)


def test_conflicting_realization_slurm_memory(tmpdir):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("REALIZATION_MEMORY 10Mb\n")
        f.write("QUEUE_SYSTEM SLURM\n")
        f.write("QUEUE_OPTION SLURM MEMORY 20M\n")

    with pytest.raises(ConfigValidationError), pytest.warns(
        ConfigWarning, match="deprecated"
    ):
        ErtConfig.from_file(filename)


def test_conflicting_realization_slurm_memory_per_cpu(tmpdir):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("REALIZATION_MEMORY 10Mb\n")
        f.write("QUEUE_SYSTEM SLURM\n")
        f.write("QUEUE_OPTION SLURM MEMORY_PER_CPU 20M\n")

    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file(filename)


def test_conflicting_realization_openpbs_memory_per_job(tmpdir):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("REALIZATION_MEMORY 10Mb\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write("QUEUE_OPTION TORQUE MEMORY_PER_JOB 20mb\n")

    with pytest.raises(ConfigValidationError), pytest.warns(
        ConfigWarning, match="deprecated"
    ):
        ErtConfig.from_file(filename)


def test_conflicting_realization_openpbs_memory_per_job_but_slurm_activated_only_warns(
    tmpdir,
):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("REALIZATION_MEMORY 10Mb\n")
        f.write("QUEUE_SYSTEM SLURM\n")
        f.write("QUEUE_OPTION TORQUE MEMORY_PER_JOB 20mb\n")

    with pytest.warns(ConfigWarning):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize("torque_memory_with_unit_str", ["gb", "mb", "1 gb"])
def test_that_invalid_memory_pr_job_raises_validation_error(
    torque_memory_with_unit_str, tmpdir
):
    filename = tmpdir / "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {torque_memory_with_unit_str}")
    with pytest.raises(ConfigValidationError), pytest.warns(
        ConfigWarning, match="deprecated"
    ):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "queue_system, queue_system_option",
    [("LSF", "LSF_QUEUE"), ("SLURM", "SQUEUE"), ("TORQUE", "QUEUE")],
)
def test_that_overwriting_QUEUE_OPTIONS_warns(
    tmp_path, monkeypatch, queue_system, queue_system_option, caplog
):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {queue_system_option} test_1\n")
        f.write(f"QUEUE_OPTION {queue_system} MAX_RUNNING 10\n")
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text(
        "JOB_SCRIPT job_dispatch.py\n"
        f"QUEUE_SYSTEM {queue_system}\n"
        f"QUEUE_OPTION {queue_system} {queue_system_option} test_0\n"
        f"QUEUE_OPTION {queue_system} MAX_RUNNING 10\n"
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    with caplog.at_level(logging.INFO):
        ErtConfig.from_file(filename)
    assert (
        f"Overwriting QUEUE_OPTION {queue_system} {queue_system_option}: \n Old value:"
        " test_0 \n New value: test_1"
    ) in caplog.text and (
        f"Overwriting QUEUE_OPTION {queue_system} MAX_RUNNING: \n Old value:"
        " 10 \n New value: 10"
    ) not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "queue_system, queue_system_option",
    [("LSF", "LSF_QUEUE"), ("SLURM", "SQUEUE")],
)
def test_initializing_empty_config_queue_options_resets_to_default_value(
    queue_system, queue_system_option
):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {queue_system_option}\n")
        f.write(f"QUEUE_OPTION {queue_system} MAX_RUNNING\n")
    config_object = ErtConfig.from_file(filename)

    if queue_system == "LSF":
        assert config_object.queue_config.queue_options.lsf_queue is None
    if queue_system == "SLURM":
        assert config_object.queue_config.queue_options.squeue == "squeue"
    assert config_object.queue_config.queue_options.max_running == 0


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "queue_system, queue_option, queue_value, err_msg",
    [
        ("SLURM", "SQUEUE_TIMEOUT", "5a", "should be a valid number"),
        ("TORQUE", "NUM_NODES", "3.5", "should be a valid integer"),
    ],
)
def test_wrong_config_option_types(queue_system, queue_option, queue_value, err_msg):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {queue_option} {queue_value}\n")

    with pytest.raises(ConfigValidationError, match=err_msg):
        if queue_system == "TORQUE":
            with pytest.warns(ConfigWarning, match="deprecated"):
                ErtConfig.from_file(filename)
        else:
            ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_configuring_another_queue_system_gives_warning():
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM LSF\n")
        f.write("QUEUE_OPTION SLURM SQUEUE_TIMEOUT ert\n")

    with pytest.warns(ConfigWarning, match="should be a valid number"):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_slurm_queue_mem_options_are_validated():
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM SLURM\n")
        f.write("QUEUE_OPTION SLURM MEMORY_PER_CPU 5mb\n")

    with pytest.raises(ConfigValidationError) as e:
        ErtConfig.from_file(filename)

    info = e.value.errors[0]

    assert "Value error, wrong memory format. Got input '5mb'." in info.message
    assert info.line == 3
    assert info.column == 35
    assert info.end_column == info.column + 3


@pytest.mark.parametrize(
    "mem_per_job",
    ["5gb", "5mb", "5kb"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_valid_torque_queue_mem_options_are_ok(mem_per_job):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM SLURM\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {mem_per_job}\n")

    with pytest.warns(ConfigWarning, match="deprecated"):
        ErtConfig.from_file(filename)


@pytest.mark.parametrize(
    "mem_per_job",
    ["5", "5g"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_torque_queue_mem_options_are_corrected(mem_per_job: str):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {mem_per_job}\n")

    with pytest.raises(ConfigValidationError) as e, pytest.warns(
        ConfigWarning, match="deprecated"
    ):
        ErtConfig.from_file(filename)

    info = e.value.errors[0]

    assert (
        f"Value error, wrong memory format. Got input '{mem_per_job}'." in info.message
    )
    assert info.line == 3
    assert info.column == 36
    assert info.end_column == info.column + len(mem_per_job)


def test_max_running_property(tmp_path):
    config_path = tmp_path / "config.ert"
    config_path.write_text(
        "NUM_REALIZATIONS 1\n"
        "QUEUE_SYSTEM TORQUE\n"
        "QUEUE_OPTION TORQUE MAX_RUNNING 17\n"
        "QUEUE_OPTION TORQUE MAX_RUNNING 19\n"
        "QUEUE_OPTION LOCAL MAX_RUNNING 11\n"
        "QUEUE_OPTION LOCAL MAX_RUNNING 13\n"
    )
    config = ErtConfig.from_file(config_path)

    assert config.queue_config.queue_system == QueueSystem.TORQUE
    assert config.queue_config.max_running == 19


@pytest.mark.parametrize("queue_system", ["LSF", "GENERIC"])
def test_multiple_submit_sleep_keywords(tmp_path, queue_system):
    config_path = tmp_path / "config.ert"
    config_path.write_text(
        "NUM_REALIZATIONS 1\n"
        "QUEUE_SYSTEM LSF\n"
        "QUEUE_OPTION LSF SUBMIT_SLEEP 10\n"
        f"QUEUE_OPTION {queue_system} SUBMIT_SLEEP 42\n"
        "QUEUE_OPTION TORQUE SUBMIT_SLEEP 22\n"
    )
    config = ErtConfig.from_file(config_path)
    assert config.queue_config.submit_sleep == 42


def test_multiple_max_submit_keywords(tmp_path):
    config_path = tmp_path / "config.ert"
    config_path.write_text("NUM_REALIZATIONS 1\nMAX_SUBMIT 10\nMAX_SUBMIT 42\n")
    config = ErtConfig.from_file(config_path)
    assert config.queue_config.max_submit == 42


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "max_submit_value, error_msg",
    [
        (-1, "must have a positive integer value as argument"),
        (0, "must have a positive integer value as argument"),
        (1.5, "must have an integer value as argument"),
    ],
)
def test_wrong_max_submit_raises_validation_error(max_submit_value, error_msg):
    with open("file.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"MAX_SUBMIT {max_submit_value}\n")
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file("file.ert")


@pytest.mark.usefixtures("use_tmpdir")
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
    def _check_results():
        ert_config = ErtConfig.from_file("config.ert")
        if key == "MAX_RUNNING":
            assert ert_config.queue_config.max_running == value
        elif key == "SUBMIT_SLEEP":
            assert ert_config.queue_config.submit_sleep == value
        else:
            raise KeyError("Unexpected key")

    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {key} 10\n")
        f.write(f"QUEUE_OPTION GENERIC {key} {value}\n")
    _check_results()

    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"{key} {value}\n")
    _check_results()


@pytest.mark.usefixtures("use_tmpdir")
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
    def _check_results():
        ert_config = ErtConfig.from_file("config.ert")
        if key == "MAX_RUNNING":
            assert ert_config.queue_config.max_running == value
        elif key == "SUBMIT_SLEEP":
            assert ert_config.queue_config.submit_sleep == value
        else:
            raise KeyError("Unexpected key")

    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {key} {value}\n")
        f.write(f"{key} {value + 42}\n")
    _check_results()

    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION GENERIC {key} {value}\n")
        f.write(f"{key} {value + 42}\n")
    _check_results()


@pytest.mark.usefixtures("use_tmpdir")
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
    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION GENERIC {key} {value}\n")
    with pytest.raises(
        ConfigValidationError, match="Input should be greater than or equal to 0"
    ):
        ErtConfig.from_file("config.ert")

    with open("config.ert", mode="w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"{key} {value}\n")
    error_msg = (
        "must have a positive integer value"
        if key == "MAX_RUNNING"
        else "Input should be greater than or equal to 0"
    )
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file("config.ert")


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
