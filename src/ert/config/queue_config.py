from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, no_type_check

from .parsing import ConfigDict, ConfigValidationError
from .queue_driver_enum import QueueDriverEnum


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 2
    queue_system: QueueDriverEnum = QueueDriverEnum.NULL_DRIVER  # type: ignore
    queue_options: Dict[QueueDriverEnum, List[Union[Tuple[str, str], str]]] = field(
        default_factory=dict
    )

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
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
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("job_dispatch.py") or "job_dispatch.py"
        )
        job_script = job_script or "job_dispatch.py"
        max_submit: int = config_dict.get("MAX_SUBMIT", 2)
        queue_options: Dict[
            QueueDriverEnum, List[Union[Tuple[str, str], str]]
        ] = defaultdict(list)
        for driver, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            queue_driver_type = QueueDriverEnum.from_string(driver + "_DRIVER")
            if values:
                queue_options[queue_driver_type].append((option_name, values[0]))
            else:
                queue_options[queue_driver_type].append(option_name)
        return QueueConfig(job_script, max_submit, queue_system, queue_options)

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueDriverEnum.LOCAL_DRIVER,  # type: ignore
            self.queue_options,
        )
