import os

import pytest

from ert.config import QueueConfig, QueueSystem
from ert.scheduler import Driver


@pytest.mark.xfail(reason="Needs reimplementation")
def test_get_driver_name():
    queue_config = QueueConfig(queue_system=QueueSystem.LOCAL)
    assert Driver.create_driver(queue_config).name == "LOCAL"
    queue_config = QueueConfig(queue_system=QueueSystem.SLURM)
    assert Driver.create_driver(queue_config).name == "SLURM"
    queue_config = QueueConfig(queue_system=QueueSystem.TORQUE)
    assert Driver.create_driver(queue_config).name == "TORQUE"
    queue_config = QueueConfig(queue_system=QueueSystem.LSF)
    assert Driver.create_driver(queue_config).name == "LSF"


@pytest.mark.xfail(reason="Needs reimplementation")
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

    assert driver.options["SBATCH"] == "/path/to/sbatch"
    assert driver.options["SCONTROL"] == "scontrol"
    assert driver.name == "SLURM"
