from __future__ import annotations

import logging
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple, no_type_check

from ert import _clib

from .parsing import (
    ConfigDict,
    ConfigValidationError,
    ConfigWarning,
    MaybeWithContext,
    QueueSystem,
)

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
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 2
    queue_system: QueueSystem = QueueSystem.LOCAL
    queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = field(
        default_factory=dict
    )

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
        selected_queue_system = QueueSystem(
            config_dict.get("QUEUE_SYSTEM", QueueSystem.LOCAL)
        )
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("job_dispatch.py") or "job_dispatch.py"
        )
        job_script = job_script or "job_dispatch.py"
        max_submit: int = config_dict.get("MAX_SUBMIT", 2)
        queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = defaultdict(list)
        for queue_system, option_name, *values in config_dict.get("QUEUE_OPTION", []):
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

        for queue_system_val in queue_options:
            if queue_options[queue_system_val]:
                _validate_queue_driver_settings(
                    queue_options[queue_system_val],
                    QueueSystem(queue_system_val).name,
                    throw_error=(queue_system_val == selected_queue_system),
                )

        if (
            selected_queue_system != QueueSystem.LOCAL
            and queue_options[selected_queue_system]
        ):
            _check_for_overwritten_queue_system_options(
                selected_queue_system,
                queue_options[selected_queue_system],
            )

        return QueueConfig(job_script, max_submit, selected_queue_system, queue_options)

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueSystem.LOCAL,
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


queue_memory_options: Mapping[str, List[str]] = {
    "LSF": [],
    "SLURM": ["MEMORY_PER_CPU", "MEMORY"],
    "TORQUE": ["MEMORY_PER_JOB"],
    "LOCAL": [],
}


@dataclass
class QueueMemoryStringFormat:
    suffixes: List[str]

    def validate(self, mem_str_format: str) -> bool:
        return (
            re.match(
                r"\d+(" + "|".join(self.suffixes) + ")$",
                mem_str_format,
            )
            is not None
        )


def parse_slurm_memopt(s: str) -> str:
    return s.lower().replace("b", "").upper()


def parse_torque_memopt(s: str) -> str:
    if re.match(r"\d+[kgmt](?!\w)", s, re.IGNORECASE):
        return s.lower() + "b"
    if re.match(r"^\d+$", s):
        return s + "kb"

    return s


queue_memory_usage_formats: Mapping[str, QueueMemoryStringFormat] = {
    "SLURM": QueueMemoryStringFormat(suffixes=["", "K", "M", "G", "T"]),
    "TORQUE": QueueMemoryStringFormat(suffixes=["kb", "mb", "gb", "KB", "MB", "GB"]),
}

queue_string_options: Mapping[str, List[str]] = {
    "LSF": [
        "LSF_RESOURCE",
        "LSF_SERVER",
        "LSF_QUEUE",
        "LSF_LOGIN_SHELL",
        "LSF_RSH_CMD",
        "BSUB_CMD",
        "BJOBS_CMD",
        "BKILL_CMD",
        "BHIST_CMD",
        "EXCLUDE_HOST",
        "PROJECT_CODE",
    ],
    "SLURM": [
        "SBATCH",
        "SCANCEL",
        "SCONTROL",
        "SQUEUE",
        "PARTITION",
        "INCLUDE_HOST",
        "EXCLUDE_HOST",
    ],
    "TORQUE": [
        "QSUB_CMD",
        "QSTAT_CMD",
        "QDEL_CMD",
        "QSTAT_OPTIONS",
        "QUEUE",
        "CLUSTER_LABEL",
        "JOB_PREFIX",
        "DEBUG_OUTPUT",
    ],
    "LOCAL": [],
}

queue_positive_int_options: Mapping[str, List[str]] = {
    "LSF": [
        "BJOBS_TIMEOUT",
        "MAX_RUNNING",
    ],
    "SLURM": [
        "MAX_RUNNING",
    ],
    "TORQUE": [
        "NUM_NODES",
        "NUM_CPUS_PER_NODE",
        "MAX_RUNNING",
    ],
    "LOCAL": ["MAX_RUNNING"],
}

queue_positive_number_options: Mapping[str, List[str]] = {
    "LSF": [
        "SUBMIT_SLEEP",
    ],
    "SLURM": [
        "SQUEUE_TIMEOUT",
        "MAX_RUNTIME",
    ],
    "TORQUE": ["SUBMIT_SLEEP", "QUEUE_QUERY_TIMEOUT"],
    "LOCAL": [],
}

queue_bool_options: Mapping[str, List[str]] = {
    "LSF": ["DEBUG_OUTPUT"],
    "SLURM": [],
    "TORQUE": ["KEEP_QSUB_OUTPUT"],
    "LOCAL": [],
}


def throw_error_or_warning(
    error_msg: str, option_value: MaybeWithContext, throw_error: bool
) -> None:
    if throw_error:
        raise ConfigValidationError.with_context(
            error_msg,
            option_value,
        )
    else:
        warnings.warn(
            ConfigWarning.with_context(
                error_msg,
                option_value,
            ),
            stacklevel=1,
        )


def _validate_queue_driver_settings(
    queue_system_options: List[Tuple[str, str]], queue_type: str, throw_error: bool
) -> None:
    for option_name, option_value in queue_system_options:
        if option_value == "":  # This is equivalent to the option not being set
            continue
        elif option_name in queue_memory_options[queue_type]:
            option_format = queue_memory_usage_formats[queue_type]

            if not option_format.validate(str(option_value)):
                throw_error_or_warning(
                    f"'{option_value}' for {option_name} is not a valid string type.",
                    option_value,
                    throw_error,
                )
        elif option_name in queue_string_options[queue_type] and not isinstance(
            option_value, str
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid string type.",
                option_value,
                throw_error,
            )

        elif option_name in queue_positive_number_options[queue_type] and (
            re.match(r"^\d+(\.\d+)?$", str(option_value)) is None
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid integer or float.",
                option_value,
                throw_error,
            )

        elif option_name in queue_positive_int_options[queue_type] and (
            re.match(r"^\d+$", str(option_value)) is None
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid positive integer.",
                option_value,
                throw_error,
            )

        elif option_name in queue_bool_options[queue_type] and not str(
            option_value
        ) in [
            "TRUE",
            "FALSE",
            "0",
            "1",
            "T",
            "F",
            "True",
            "False",
        ]:
            throw_error_or_warning(
                f"The '{option_value}' for {option_name} should be either TRUE or FALSE.",
                option_value,
                throw_error,
            )
