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
from ert.job_queue import Driver


def test_create_local_copy_is_a_copy_with_local_queue_system():
    queue_config = QueueConfig(queue_system=QueueSystem.LSF)
    assert queue_config.queue_system == QueueSystem.LSF
    assert queue_config.create_local_copy().queue_system == QueueSystem.LOCAL


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_that_default_max_running_is_unlimited(num_real):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS {num_real}\nQUEUE_SYSTEM SLURM\n")
    # max_running == 0 means unlimited
    assert (
        Driver.create_driver(
            ErtConfig.from_file(filename).queue_config
        ).get_max_running()
        == 0
    )


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_that_an_invalid_queue_system_provided_raises_validation_error(num_real):
    filename = "config.ert"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS {num_real}\nQUEUE_SYSTEM VOID\n")

    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="Invalid QUEUE_SYSTEM provided: 'VOID'",
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
@given(memory_with_unit())
def test_torque_queue_config_memory_pr_job(memory_with_unit_str):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {memory_with_unit_str}")

    config = ErtConfig.from_file(filename)

    driver = Driver.create_driver(config.queue_config)

    assert driver.get_option("MEMORY_PER_JOB") == memory_with_unit_str


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize("memory_with_unit_str", ["1", "gb", "mb", "1 gb", "1kb"])
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
    "queue_system, queue_system_option",
    [("LSF", "LSF_SERVER"), ("SLURM", "SQUEUE"), ("TORQUE", "QUEUE")],
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
    driver = Driver.create_driver(config_object.queue_config)
    assert driver.get_option(queue_system_option) == ""
    assert driver.get_option("MAX_RUNNING") == "0"
    for options in config_object.queue_config.queue_options[queue_system]:
        assert isinstance(options, tuple)


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
