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
from ert.scheduler import Driver


def test_create_local_copy_is_a_copy_with_local_queue_system():
    queue_config = QueueConfig(queue_system=QueueSystem.LSF)
    assert queue_config.queue_system == QueueSystem.LSF
    assert queue_config.create_local_copy().queue_system == QueueSystem.LOCAL


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
    unit = draw(st.sampled_from(["gb", "mb"]))
    return f"{memory_value}{unit}"


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize("memory_with_unit_str", ["gb", "mb", "1 gb"])
def test_that_invalid_memory_pr_job_raises_validation_error(memory_with_unit_str):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {memory_with_unit_str}")
    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "queue_system, queue_system_option",
    [("LSF", "LSF_SERVER"), ("SLURM", "SQUEUE"), ("TORQUE", "QUEUE")],
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
        " test_0 \n New value: test_1" in caplog.text
        and f"Overwriting QUEUE_OPTION {queue_system} MAX_RUNNING: \n Old value:"
        " 10 \n New value: 10" not in caplog.text
    )


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_undefined_LSF_SERVER_environment_variable_raises_validation_error():
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM LSF\n")
        f.write("QUEUE_OPTION LSF LSF_SERVER $MY_SERVER\n")
    with pytest.raises(
        ConfigValidationError,
        match=(
            r"Invalid server name specified for QUEUE_OPTION LSF LSF_SERVER: "
            r"\$MY_SERVER. Server name is currently an undefined environment variable."
            r" The LSF_SERVER keyword is usually provided by the site-configuration"
            r" file, beware that you are effectively replacing the default value"
            r" provided."
        ),
    ):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "queue_system, queue_option, queue_value, err_msg",
    [
        ("LSF", "BJOBS_TIMEOUT", "-3", "is not a valid positive integer"),
        ("SLURM", "SQUEUE_TIMEOUT", "5a", "is not a valid integer or float"),
        ("TORQUE", "NUM_NODES", "3.5", "is not a valid positive integer"),
    ],
)
def test_wrong_config_option_types(queue_system, queue_option, queue_value, err_msg):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write(f"QUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {queue_option} {queue_value}\n")

    with pytest.raises(ConfigValidationError, match=err_msg):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_configuring_another_queue_system_gives_warning():
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM LSF\n")
        f.write("QUEUE_OPTION SLURM SQUEUE_TIMEOUT ert\n")

    with pytest.warns(ConfigWarning, match="is not a valid integer or float"):
        ErtConfig.from_file(filename)


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

    ErtConfig.from_file(filename)


@pytest.mark.parametrize(
    "mem_per_job",
    ["5", "5g"],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_torque_queue_mem_options_are_corrected(mem_per_job):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {mem_per_job}\n")

    with pytest.raises(ConfigValidationError) as e:
        ert_config = ErtConfig.from_file(filename)
        torque_opts = ert_config.queue_config.queue_options[QueueSystem.TORQUE]
        assert torque_opts[0][1] == mem_per_job

    info = e.value.errors[0]

    assert f"'{mem_per_job}' for MEMORY_PER_JOB is not a valid" in info.message
    assert info.line == 3
    assert info.column == 36
    assert info.end_column == info.column + len(mem_per_job)
