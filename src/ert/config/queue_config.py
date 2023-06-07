from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, no_type_check

from .parsing import ConfigDict, ConfigValidationError, ErrorInfo
from .queue_system import QueueSystem


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 2
    queue_system: QueueSystem = QueueSystem.NULL  # type: ignore
    queue_options: Dict[QueueSystem, List[Union[Tuple[str, str], str]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        errors = []
        for _, value in [
            setting
            for settings in self.queue_options.values()
            for setting in settings
            if isinstance(setting, tuple) and setting[0] == "MAX_RUNNING"
        ]:
            err_msg = "QUEUE_OPTION MAX_RUNNING is"
            try:
                int_val = int(value)
                if int_val < 0:
                    errors.append(
                        ErrorInfo(f"{err_msg} negative: {str(value)!r}").set_context(
                            value
                        )
                    )
            except ValueError:
                errors.append(
                    ErrorInfo(f"{err_msg} not an integer: {str(value)!r}").set_context(
                        value
                    )
                )
        if errors:
            raise ConfigValidationError.from_collected(errors)

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
        queue_system = config_dict.get("QUEUE_SYSTEM", "LOCAL")

        valid_queue_systems = []

        for driver_names in QueueSystem.enums():
            if driver_names.name not in str(QueueSystem.NULL):
                valid_queue_systems.append(driver_names.name)

        if queue_system not in valid_queue_systems:
            raise ConfigValidationError(
                f"Invalid QUEUE_SYSTEM provided: {queue_system!r}. Valid choices for "
                f"QUEUE_SYSTEM are {valid_queue_systems!r}"
            )

        queue_system = QueueSystem.from_string(queue_system)
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("job_dispatch.py") or "job_dispatch.py"
        )
        job_script = job_script or "job_dispatch.py"
        max_submit: int = config_dict.get("MAX_SUBMIT", 2)
        queue_options: Dict[
            QueueSystem, List[Union[Tuple[str, str], str]]
        ] = defaultdict(list)
        for system, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            queue_driver_type = QueueSystem.from_string(system)
            if values:
                queue_options[queue_driver_type].append((option_name, values[0]))
            else:
                queue_options[queue_driver_type].append(option_name)

        if queue_system == QueueSystem.TORQUE and queue_options[QueueSystem.TORQUE]:
            _validate_torque_options(queue_options[QueueSystem.TORQUE])

        return QueueConfig(job_script, max_submit, queue_system, queue_options)

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueSystem.LOCAL,  # type: ignore
            self.queue_options,
        )


def _validate_torque_options(torque_options: List[Tuple[str, str]]) -> None:
    for option_strings in torque_options:
        if isinstance(option_strings, tuple):
            option_name = option_strings[0]
            option_value = option_strings[1]
            if (
                option_value != ""  # This is equivalent to the option not being set
                and option_name == "MEMORY_PER_JOB"
                and re.match("[0-9]+[mg]b", option_value) is None
            ):
                raise ConfigValidationError(
                    f"The value '{option_value}' is not valid for the Torque option "
                    "MEMORY_PER_JOB, it must be of "
                    "the format '<integer>mb' or '<integer>gb'."
                )
