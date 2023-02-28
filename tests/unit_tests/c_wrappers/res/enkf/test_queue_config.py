import os
import os.path
import stat
from pathlib import Path

from ert._c_wrappers.enkf import QueueConfig
from ert._c_wrappers.job_queue import QueueDriverEnum


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config.create_job_queue()
    queue_config_copy = queue_config.create_local_copy()
    assert queue_config_copy.queue_system == QueueDriverEnum.LOCAL_DRIVER


def test_queue_config_constructor(minimum_case):
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
    minimum_queue_config = minimum_case.resConfig().queue_config

    # Depends on where you run the tests
    assert minimum_queue_config in (queue_config_absolute, queue_config_relative)


def test_set_and_unset_option():
    queue_config = QueueConfig(
        job_script="script.sh",
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
                ("MAX_RUNNING", "50"),
                "MAX_RUNNING",
            ]
        },
    )
    assert queue_config.create_driver().get_option("MAX_RUNNING") == "0"


def test_get_slurm_queue_config(setup_case):
    ert_config = setup_case("simple_config", "slurm_config")
    queue_config = ert_config.queue_config

    assert queue_config.queue_system == QueueDriverEnum.SLURM_DRIVER
    driver = queue_config.create_driver()
    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    assert driver.name == "SLURM"
