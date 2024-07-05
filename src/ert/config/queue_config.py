from __future__ import annotations

import logging
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple, no_type_check

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    MaybeWithContext,
    QueueSystem,
)

GENERIC_QUEUE_OPTIONS: List[str] = ["MAX_RUNNING", "SUBMIT_SLEEP"]
LSF_DRIVER_OPTIONS = [
    "BHIST_CMD",
    "BJOBS_CMD",
    "BJOBS_TIMEOUT",
    "BKILL_CMD",
    "BSUB_CMD",
    "DEBUG_OUTPUT",
    "EXCLUDE_HOST",
    "LSF_LOGIN_SHELL",
    "LSF_QUEUE",
    "LSF_RESOURCE",
    "LSF_RSH_CMD",
    "LSF_SERVER",
    "PROJECT_CODE",
]
OPENPBS_DRIVER_OPTIONS: List[str] = [
    "CLUSTER_LABEL",
    "DEBUG_OUTPUT",
    "JOB_PREFIX",
    "KEEP_QSUB_OUTPUT",
    "MEMORY_PER_JOB",
    "NUM_CPUS_PER_NODE",
    "NUM_NODES",
    "PROJECT_CODE",
    "QDEL_CMD",
    "QSTAT_CMD",
    "QSTAT_OPTIONS",
    "QSUB_CMD",
    "QUEUE",
    "QUEUE_QUERY_TIMEOUT",
]
SLURM_DRIVER_OPTIONS: List[str] = [
    "EXCLUDE_HOST",
    "INCLUDE_HOST",
    "MEMORY",
    "MEMORY_PER_CPU",
    "PARTITION",
    "PROJECT_CODE",
    "SBATCH",
    "SCANCEL",
    "SCONTROL",
    "SQUEUE",
    "SQUEUE_TIMEOUT",
]
VALID_QUEUE_OPTIONS: Dict[Any, List[str]] = {
    QueueSystem.LOCAL: [] + GENERIC_QUEUE_OPTIONS,  # No specific options in driver
    QueueSystem.LSF: LSF_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.SLURM: SLURM_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.TORQUE: OPENPBS_DRIVER_OPTIONS + GENERIC_QUEUE_OPTIONS,
    QueueSystem.GENERIC: GENERIC_QUEUE_OPTIONS,
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
        "PROJECT_CODE",
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
        "PROJECT_CODE",
    ],
    "LOCAL": ["PROJECT_CODE"],
    "GENERIC": [],
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
    "GENERIC": ["MAX_RUNNING"],
}
queue_positive_number_options: Mapping[str, List[str]] = {
    "LSF": ["SUBMIT_SLEEP"],
    "SLURM": ["SUBMIT_SLEEP", "SQUEUE_TIMEOUT", "MAX_RUNTIME"],
    "TORQUE": ["SUBMIT_SLEEP", "QUEUE_QUERY_TIMEOUT"],
    "LOCAL": ["SUBMIT_SLEEP"],
    "GENERIC": ["SUBMIT_SLEEP"],
}
queue_bool_options: Mapping[str, List[str]] = {
    "LSF": ["DEBUG_OUTPUT"],
    "SLURM": [],
    "TORQUE": ["KEEP_QSUB_OUTPUT"],
    "LOCAL": [],
    "GENERIC": [],
}
queue_memory_options: Mapping[str, List[str]] = {
    "LSF": [],
    "SLURM": ["MEMORY_PER_CPU", "MEMORY"],
    "TORQUE": ["MEMORY_PER_JOB"],
    "LOCAL": [],
    "GENERIC": [],
}


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    max_submit: int = 1
    queue_system: QueueSystem = QueueSystem.LOCAL
    queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = field(
        default_factory=dict
    )
    stop_long_running: bool = False

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
        selected_queue_system = QueueSystem(
            config_dict.get("QUEUE_SYSTEM", QueueSystem.LOCAL)
        )
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("job_dispatch.py") or "job_dispatch.py"
        )
        max_submit: int = config_dict.get(ConfigKeys.MAX_SUBMIT, 1)
        stop_long_running = config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False)
        queue_options: Dict[QueueSystem, List[Tuple[str, str]]] = defaultdict(list)
        for queue_system, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            if queue_system == QueueSystem.GENERIC:
                queue_system = selected_queue_system
            if option_name not in VALID_QUEUE_OPTIONS[queue_system]:
                raise ConfigValidationError(
                    f"Invalid QUEUE_OPTION for {queue_system.name}: '{option_name}'. "
                    f"Valid choices are {sorted(VALID_QUEUE_OPTIONS[queue_system])}."
                )

            queue_options[queue_system].append(
                (option_name, values[0] if values else "")
            )
        queue_options[selected_queue_system] = _add_generic_queue_options(
            config_dict, queue_options[selected_queue_system]
        )

        if "PROJECT_CODE" not in _option_list_to_dict(
            queue_options[selected_queue_system]
        ):
            tags = {
                fm_name.lower()
                for fm_name, *_ in config_dict.get(ConfigKeys.FORWARD_MODEL, [])
                if fm_name in ["RMS", "FLOW", "ECLIPSE100", "ECLIPSE300"]
            }
            if tags:
                queue_options[selected_queue_system].append(
                    ("PROJECT_CODE", "+".join(tags))
                )
        _check_queue_option_settings(selected_queue_system, queue_options, config_dict)
        return QueueConfig(
            job_script,
            max_submit,
            selected_queue_system,
            queue_options,
            stop_long_running=stop_long_running,
        )

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.max_submit,
            QueueSystem.LOCAL,
            self.queue_options,
            stop_long_running=self.stop_long_running,
        )

    @property
    def max_running(self) -> int:
        max_running = 0
        for key, val in self.queue_options.get(self.queue_system, []):
            if key == "MAX_RUNNING":
                max_running = int(val)
        return max_running

    @property
    def submit_sleep(self) -> float:
        return float(
            dict(self.queue_options.get(self.queue_system, [])).get("SUBMIT_SLEEP", 0.0)
        )

    @property
    def queue_options_as_dict(self) -> Dict[str, str]:
        return dict(self.queue_options.get(self.queue_system, []))


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


@no_type_check
def _check_queue_option_settings(
    selected_queue_system: QueueSystem,
    queue_options: List[Tuple[str, str]],
    config_dict: ConfigDict,
) -> None:
    for queue_system, queue_system_settings in queue_options.items():
        if queue_system_settings:
            _validate_queue_driver_settings(
                queue_system_settings,
                QueueSystem(queue_system).name,
                throw_error=(queue_system == selected_queue_system),
            )

    if (
        selected_queue_system != QueueSystem.LOCAL
        and queue_options[selected_queue_system]
    ):
        _check_for_overwritten_queue_system_options(
            selected_queue_system,
            queue_options[selected_queue_system],
        )
    if selected_queue_system == QueueSystem.TORQUE:
        _check_num_cpu_requirement(
            config_dict.get("NUM_CPU", 1),
            queue_options[selected_queue_system],
        )


def _option_list_to_dict(option_list: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    temp_dict: Dict[str, List[str]] = defaultdict(list)
    for option_string in option_list:
        temp_dict[option_string[0]].append(option_string[1])
    return temp_dict


def _check_for_overwritten_queue_system_options(
    selected_queue_system: QueueSystem,
    queue_system_options: List[Tuple[str, str]],
) -> None:
    for option_name, option_values in _option_list_to_dict(
        queue_system_options
    ).items():
        if len(option_values) > 1 and option_values[0] != option_values[-1]:
            logging.info(
                f"Overwriting QUEUE_OPTION {selected_queue_system} {option_name}:"
                f" \n Old value: {option_values[0]} \n New value: {option_values[-1]}"
            )


def _check_num_cpu_requirement(
    num_cpu: int,
    queue_system_options: List[Tuple[str, str]],
) -> None:
    torque_options = _option_list_to_dict(queue_system_options)
    num_nodes_str = torque_options.get("NUM_NODES", [""])[-1]
    num_cpus_per_node_str = torque_options.get("NUM_CPUS_PER_NODE", [""])[-1]
    if not num_nodes_str and not num_cpus_per_node_str:
        # NUM_CPU will take precedence when deprecated options are not set
        return
    num_nodes = int(num_nodes_str) if num_nodes_str else 1
    num_cpus_per_node = int(num_cpus_per_node_str) if num_cpus_per_node_str else 1
    if num_cpu != num_nodes * num_cpus_per_node:
        raise ConfigValidationError(
            f"When NUM_CPU is {num_cpu}, then the product of NUM_NODES ({num_nodes}) "
            f"and NUM_CPUS_PER_NODE ({num_cpus_per_node}) must be equal."
        )


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
        if not option_value:  # This is equivalent to the option not being set
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


@no_type_check
def _add_generic_queue_options(
    config_dict: ConfigDict, queue_options: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    options_dict = _option_list_to_dict(queue_options)
    for generic_option in GENERIC_QUEUE_OPTIONS:
        value = config_dict.get(generic_option, None)
        if generic_option not in options_dict and value is not None:
            queue_options.append((generic_option, value))
    return queue_options
