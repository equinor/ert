from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ert import _clib

from ._config_values import (
    DEFAULT_JOB_SCRIPT,
    DEFAULT_MAX_SUBMIT,
    DEFAULT_QUEUE_SYSTEM,
    ErtConfigValues,
)
from .parsing import ConfigValidationError, ErrorInfo
from .queue_system import QueueSystem

GENERIC_QUEUE_OPTIONS: List[str] = ["MAX_RUNNING"]
VALID_QUEUE_OPTIONS: Dict[Any, List[str]] = {
    QueueSystem.TORQUE: _clib.torque_driver.TORQUE_DRIVER_OPTIONS
    + GENERIC_QUEUE_OPTIONS,
    QueueSystem.LOCAL: [] + GENERIC_QUEUE_OPTIONS,  # No specific options in driver
    QueueSystem.SLURM: _clib.slurm_driver.SLURM_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.LSF: _clib.lsf_driver.LSF_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
}


@dataclass
class QueueConfig:
    job_script: str = DEFAULT_JOB_SCRIPT
    max_submit: int = DEFAULT_MAX_SUBMIT
    queue_system: QueueSystem = DEFAULT_QUEUE_SYSTEM  # type: ignore
    queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        errors = []
        for _, value in [
            setting
            for settings in self.queue_options.values()
            for setting in settings
            if setting[0] == "MAX_RUNNING" and setting[1]
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

    @classmethod
    def from_values(cls, config_values: ErtConfigValues) -> QueueConfig:
        queue_system = config_values.queue_system
        job_script: str = config_values.job_script
        max_submit: int = config_values.max_submit
        queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = defaultdict(list)
        for queue_system, option_name, *values in config_values.queue_option:
            if option_name not in VALID_QUEUE_OPTIONS[queue_system]:
                raise ConfigValidationError(
                    f"Invalid QUEUE_OPTION for {queue_system.name}: '{option_name}'. "
                    f"Valid choices are {sorted(VALID_QUEUE_OPTIONS[queue_system])}."
                )
            queue_options[queue_system].append(
                (option_name, values[0] if values else "")
            )
            if values and option_name == "LSF_SERVER" and values[0].startswith("$"):
                raise ConfigValidationError(
                    "Invalid server name specified for QUEUE_OPTION LSF"
                    f" LSF_SERVER: {values[0]}. Server name is currently an"
                    " undefined environment variable. The LSF_SERVER keyword is"
                    " usually provided by the site-configuration file, beware that"
                    " you are effectively replacing the default value provided."
                )
        if queue_system == QueueSystem.TORQUE and queue_options[QueueSystem.TORQUE]:
            _validate_torque_options(queue_options[QueueSystem.TORQUE])

        if queue_system != QueueSystem.LOCAL and queue_options[queue_system]:
            _check_for_overwritten_queue_system_options(
                queue_system,
                queue_options[queue_system],
            )

        return QueueConfig(job_script, max_submit, queue_system, queue_options)

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueSystem.LOCAL,  # type: ignore
            self.queue_options,
        )


def _check_for_overwritten_queue_system_options(
    selected_queue_system: QueueSystem,
    queue_system_options: List[Tuple[str, str]],
) -> None:
    def generate_dict(option_list: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        temp_dict: Dict[str, List[str]] = defaultdict(list)
        for option_string in option_list:
            temp_dict.setdefault(option_string[0], []).append(option_string[1])
        return temp_dict

    for option_name, option_values in generate_dict(queue_system_options).items():
        if len(option_values) > 1 and option_values[0] != option_values[-1]:
            logging.info(
                f"Overwriting QUEUE_OPTION {selected_queue_system} {option_name}:"
                f" \n Old value: {option_values[0]} \n New value: {option_values[-1]}"
            )


def _validate_torque_options(torque_options: List[Tuple[str, str]]) -> None:
    for option_strings in torque_options:
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
