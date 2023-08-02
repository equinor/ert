import os

from ert.config import QueueConfig, QueueDriverEnum
from ert.job_queue import Driver


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
    assert Driver.create_driver(queue_config).get_option("MAX_RUNNING") == "0"


def test_get_slurm_queue_config():
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueDriverEnum.SLURM_DRIVER,
        max_submit=2,
        queue_options={
            QueueDriverEnum.SLURM_DRIVER: [
                ("MAX_RUNNING", "50"),
                ("SBATCH", "/path/to/sbatch"),
                ("SQUEUE", "/path/to/squeue"),
            ]
        },
    )

    assert queue_config.queue_system == QueueDriverEnum.SLURM_DRIVER
    driver = Driver.create_driver(queue_config)
    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    assert driver.name == "SLURM"
