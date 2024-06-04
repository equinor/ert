from __future__ import annotations

import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from ert import _clib

from .parsing import (
    ConfigDict,
    ConfigValidationError,
    MaybeWithContext,
    QueueSystem,
)

GENERIC_QUEUE_OPTIONS: List[str] = ["MAX_RUNNING"]
OPENPBS_DRIVER_OPTIONS: List[str] = [
    "CLUSTER_LABEL",
    "DEBUG_OUTPUT",
    "JOB_PREFIX",
    "KEEP_QSUB_OUTPUT",
    "MEMORY_PER_JOB",
    "NUM_CPUS_PER_NODE",
    "NUM_NODES",
    "QDEL_CMD",
    "QSTAT_CMD",
    "QSTAT_OPTIONS",
    "QSUB_CMD",
    "QUEUE",
    "QUEUE_QUERY_TIMEOUT",
    "SUBMIT_SLEEP",
]
VALID_QUEUE_OPTIONS: Dict[QueueSystem, List[str]] = {
    QueueSystem.TORQUE: OPENPBS_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.LOCAL: [] + GENERIC_QUEUE_OPTIONS,  # No specific options in driver
    QueueSystem.SLURM: _clib.slurm_driver.SLURM_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.LSF: _clib.lsf_driver.LSF_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
}


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 1
    submit_sleep: float = 0.0
    queue_system: QueueSystem = QueueSystem.LOCAL
    queue_options: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
        selected_queue_system = QueueSystem(
            config_dict.get("QUEUE_SYSTEM", QueueSystem.LOCAL)
        )
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("job_dispatch.py") or "job_dispatch.py"
        )
        max_submit: int = config_dict.get("MAX_SUBMIT", 1)
        submit_sleep: float = config_dict.get("SUBMIT_SLEEP", 0.0)
        queue_options_dict: Dict[str, list] = defaultdict(list)
        for queue_system, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            if queue_system != selected_queue_system:
                continue
            if option_name not in VALID_QUEUE_OPTIONS[queue_system]:
                raise ConfigValidationError(
                    f"Invalid QUEUE_OPTION for {queue_system.name}: '{option_name}'. "
                    f"Valid choices are {sorted(VALID_QUEUE_OPTIONS[queue_system])}."
                )

            queue_options_dict[option_name].append(values[0] if values else "")
            if (
                values
                and option_name == "SUBMIT_SLEEP"
                and selected_queue_system == queue_system
            ):
                submit_sleep = float(values[0])

        if queue_options_dict:
            _check_for_overwritten_queue_system_options(
                selected_queue_system,
                queue_options_dict,
            )

            _validate_queue_driver_settings(
                queue_options_dict,
                selected_queue_system,
            )
        if selected_queue_system == QueueSystem.TORQUE:
            _check_num_cpu_requirement(
                config_dict.get("NUM_CPU", 1),
                queue_options_dict,
            )
        return QueueConfig(
            job_script,
            max_submit,
            submit_sleep,
            selected_queue_system,
            {key: value[-1] for key, value in queue_options_dict.items()},
        )

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            self.submit_sleep,
            QueueSystem.LOCAL,
            {"MAX_RUNNING": self.queue_options.get("MAX_RUNNING", 0)},
        )

    @property
    def max_running(self) -> int:
        return int(self.queue_options.get("MAX_RUNNING", 0))


def _check_for_overwritten_queue_system_options(
    selected_queue_system: QueueSystem,
    queue_system_options: Dict[str, List[str]],
) -> None:
    for option_name, option_values in queue_system_options.items():
        if len(option_values) > 1 and option_values[0] != option_values[-1]:
            logging.info(
                f"Overwriting QUEUE_OPTION {selected_queue_system} {option_name}:"
                f" \n Old value: {option_values[0]} \n New value: {option_values[-1]}"
            )


def _check_num_cpu_requirement(
    num_cpu: int,
    torque_options: Dict[str, List[str]],
) -> None:
    num_nodes_str = torque_options.get("NUM_NODES", [""])[-1]
    num_cpus_per_node_str = torque_options.get("NUM_CPUS_PER_NODE", [""])[-1]
    num_nodes = int(num_nodes_str) if num_nodes_str else 1
    num_cpus_per_node = int(num_cpus_per_node_str) if num_cpus_per_node_str else 1
    if num_cpu != num_nodes * num_cpus_per_node:
        raise ConfigValidationError(
            f"When NUM_CPU is {num_cpu}, then the product of NUM_NODES ({num_nodes}) "
            f"and NUM_CPUS_PER_NODE ({num_cpus_per_node}) must be equal."
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


def throw_error_or_warning(error_msg: str, option_value: MaybeWithContext) -> None:
    raise ConfigValidationError.with_context(
        error_msg,
        option_value,
    )


def _validate_queue_driver_settings(
    queue_system_options: Dict[str, List[str]], queue_type: QueueSystem
) -> None:
    for option_name, option_values in queue_system_options.items():
        option_value = option_values[-1]
        if not option_value:  # This is equivalent to the option not being set
            continue
        if option_name in queue_memory_options[queue_type]:
            option_format = queue_memory_usage_formats[queue_type]

            if not option_format.validate(str(option_value)):
                throw_error_or_warning(
                    f"'{option_value}' for {option_name} is not a valid string type.",
                    option_value,
                )
        elif option_name in queue_string_options[queue_type] and not isinstance(
            option_value, str
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid string type.",
                option_value,
            )

        elif option_name in queue_positive_number_options[queue_type] and (
            re.match(r"^\d+(\.\d+)?$", str(option_value)) is None
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid integer or float.",
                option_value,
            )

        elif option_name in queue_positive_int_options[queue_type] and (
            re.match(r"^\d+$", str(option_value)) is None
        ):
            throw_error_or_warning(
                f"'{option_value}' for {option_name} is not a valid positive integer.",
                option_value,
            )

        elif option_name in queue_bool_options[queue_type] and str(
            option_value
        ) not in [
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
            )
