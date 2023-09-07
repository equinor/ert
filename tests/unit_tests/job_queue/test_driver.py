import os

from ert.config import QueueConfig, QueueSystem
from ert.job_queue import Driver


def test_set_and_unset_option():
    queue_config = QueueConfig(
        job_script="script.sh",
        queue_system=QueueSystem(2),
        max_submit=2,
        queue_options={
            QueueSystem.LOCAL: [
                ("MAX_RUNNING", "50"),
                "MAX_RUNNING",
            ]
        },
    )
    driver = Driver.create_driver(queue_config)
    assert driver.get_option("MAX_RUNNING") == "0"
    assert driver.set_option("MAX_RUNNING", "42")
    assert driver.get_option("MAX_RUNNING") == "42"
    driver.unset_option("MAX_RUNNING")
    assert driver.get_option("MAX_RUNNING") == "0"


def test_get_slurm_queue_config():
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.SLURM,
        max_submit=2,
        queue_options={
            QueueSystem.SLURM: [
                ("MAX_RUNNING", "50"),
                ("SBATCH", "/path/to/sbatch"),
                ("SQUEUE", "/path/to/squeue"),
            ]
        },
    )

    assert queue_config.queue_system == QueueSystem.SLURM
    driver = Driver.create_driver(queue_config)

    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    driver.unset_option("SCONTROL")
    assert driver.get_option("SCONTROL") == ""
    assert driver.name == "SLURM"
