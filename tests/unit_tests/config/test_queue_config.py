import os
import os.path
import stat
from pathlib import Path

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


@st.composite
def memory_with_unit(draw):
    memory_value = draw(st.integers(min_value=1, max_value=10000))
    unit = draw(st.sampled_from(["gb", "mb"]))
    return f"{memory_value}{unit}"


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config_copy = queue_config.create_local_copy()
    assert queue_config_copy.queue_system == QueueSystem.LOCAL


def test_queue_config_constructor(minimum_case):
    with open(minimum_case.ert_config.user_config_file, "a", encoding="utf-8") as fout:
        fout.write("\nJOB_SCRIPT script.sh")
    Path("script.sh").write_text("", encoding="utf-8")
    current_mode = os.stat("script.sh").st_mode
    os.chmod("script.sh", current_mode | stat.S_IEXEC)
    queue_config_relative = QueueConfig(
        job_script="script.sh",
        queue_system=QueueSystem(2),
        max_submit=2,
        queue_options={
            QueueSystem.LOCAL: [
                ("MAX_RUNNING", "1"),
                ("MAX_RUNNING", "50"),
            ]
        },
    )

    queue_config_absolute = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem(2),
        max_submit=2,
        queue_options={
            QueueSystem.LOCAL: [
                ("MAX_RUNNING", "1"),
                ("MAX_RUNNING", "50"),
            ]
        },
    )
    minimum_queue_config = ErtConfig.from_file("minimum_config").queue_config

    # Depends on where you run the tests
    assert minimum_queue_config in (queue_config_absolute, queue_config_relative)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_queue_config_default_max_running_is_unlimited(num_real):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS {num_real}\nQUEUE_SYSTEM SLURM\n")
    # max_running == 0 means unlimited
    assert (
        Driver.create_driver(ErtConfig.from_file(filename).queue_config).max_running
        == 0
    )


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_queue_config_invalid_queue_system_provided(num_real):
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
def test_queue_config_invalid_queue_option_provided(queue_system, invalid_option):
    filename = "config.ert"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NUM_REALIZATIONS 1\nQUEUE_SYSTEM {queue_system}\n")
        f.write(f"QUEUE_OPTION {queue_system} {invalid_option}")

    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=f"Invalid QUEUE_OPTION for {queue_system}: '{invalid_option}'",
    ):
        _ = ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(memory_with_unit())
def test_torque_queue_config_memory_pr_job(memory_with_unit_str):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {memory_with_unit_str}")

    ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize("memory_with_unit_str", ["1", "gb", "mb", "1 gb", "1kb"])
def test_torque_queue_config_invalid_memory_pr_job(memory_with_unit_str):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM TORQUE\n")
        f.write(f"QUEUE_OPTION TORQUE MEMORY_PER_JOB {memory_with_unit_str}")
    with pytest.raises(ConfigValidationError):
        ErtConfig.from_file(filename)


@pytest.mark.usefixtures("use_tmpdir")
def test_queue_option_LSF_SERVER_set_by_user_warning(tmp_path, monkeypatch):
    filename = "config.ert"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1\n")
        f.write("QUEUE_SYSTEM LSF\n")
        f.write("QUEUE_OPTION LSF LSF_SERVER test_server_1\n")
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text(
        "JOB_SCRIPT job_dispatch.py\n"
        "QUEUE_SYSTEM LSF\n"
        "QUEUE_OPTION LSF LSF_SERVER test_server_2\n"
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))
    with pytest.warns(
        ConfigWarning,
        match=r"Overwriting LSF_SERVER keyword, this may lead to an error.",
    ):
        ErtConfig.from_file(filename)
