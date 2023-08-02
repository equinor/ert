import os
import os.path
import stat
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import ConfigValidationError, ErtConfig, QueueConfig, QueueDriverEnum
from ert.job_queue import Driver


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config_copy = queue_config.create_local_copy()
    assert queue_config_copy.queue_system == QueueDriverEnum.LOCAL_DRIVER


def test_queue_config_constructor(minimum_case):
    with open(minimum_case.ert_config.user_config_file, "a", encoding="utf-8") as fout:
        fout.write("\nJOB_SCRIPT script.sh")
    Path("script.sh").write_text("", encoding="utf-8")
    current_mode = os.stat("script.sh").st_mode
    os.chmod("script.sh", current_mode | stat.S_IEXEC)
    queue_config_relative = QueueConfig(
        job_script="script.sh",
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
                ("MAX_RUNNING", "1"),
                ("MAX_RUNNING", "50"),
            ]
        },
    )

    queue_config_absolute = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
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
