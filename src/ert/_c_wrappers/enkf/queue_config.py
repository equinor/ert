import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.job_queue import Driver, JobQueue, QueueDriverEnum


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 2
    queue_system: QueueDriverEnum = QueueDriverEnum.NULL_DRIVER
    queue_options: Dict[QueueDriverEnum, List[Union[Tuple[str, str], str]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, config_dict):
        queue_system = config_dict.get("QUEUE_SYSTEM", "LOCAL")

        valid_queue_systems = []

        for driver_names in QueueDriverEnum.enums():
            if driver_names.name not in str(QueueDriverEnum.NULL_DRIVER):
                valid_queue_systems.append(driver_names.name[: -len("_DRIVER")])

        if queue_system not in valid_queue_systems:
            raise ConfigValidationError(
                f"Invalid QUEUE_SYSTEM provided: {queue_system!r}. Valid choices for "
                f"QUEUE_SYSTEM are {valid_queue_systems!r}"
            )

        queue_system = QueueDriverEnum.from_string(f"{queue_system}_DRIVER")
        job_script = config_dict.get("JOB_SCRIPT", shutil.which("job_dispatch.py"))
        job_script = job_script or "job_dispatch.py"
        max_submit = config_dict.get("MAX_SUBMIT", 2)
        queue_options = defaultdict(list)
        for driver, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            queue_driver_type = QueueDriverEnum.from_string(driver + "_DRIVER")
            if values:
                queue_options[queue_driver_type].append((option_name, values[0]))
            else:
                queue_options[queue_driver_type].append(option_name)
        return QueueConfig(job_script, max_submit, queue_system, queue_options)

    def create_driver(self) -> Driver:
        driver = Driver(self.queue_system)
        if self.queue_system in self.queue_options:
            for setting in self.queue_options[self.queue_system]:
                if isinstance(setting, Tuple):
                    driver.set_option(*setting)
                else:
                    driver.unset_option(setting)
        return driver

    def create_job_queue(self) -> JobQueue:
        queue = JobQueue(self.create_driver(), max_submit=self.max_submit)
        return queue

    def create_local_copy(self):
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueDriverEnum.LOCAL_DRIVER,
            self.queue_options,
        )
