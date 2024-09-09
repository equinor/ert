from __future__ import annotations

import logging
import re
import shutil
from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, no_type_check

import pydantic
from typing_extensions import Annotated

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    MaybeWithContext,
    QueueSystem,
    QueueSystemWithGeneric,
)

logger = logging.getLogger(__name__)

NonEmptyString = Annotated[str, pydantic.StringConstraints(min_length=1)]


@pydantic.dataclasses.dataclass(config={"extra": "forbid", "validate_assignment": True})
class QueueOptions:
    max_running: pydantic.NonNegativeInt = 0
    submit_sleep: pydantic.NonNegativeFloat = 0.0
    project_code: Optional[str] = None

    @staticmethod
    def create_queue_options(
        queue_system: QueueSystem,
        options: Dict[str, Any],
        is_selected_queue_system: bool,
    ) -> Optional[QueueOptions]:
        lower_case_options = {key.lower(): value for key, value in options.items()}
        try:
            if queue_system == QueueSystem.LSF:
                return LsfQueueOptions(**lower_case_options)
            elif queue_system == QueueSystem.SLURM:
                return SlurmQueueOptions(**lower_case_options)
            elif queue_system == QueueSystem.TORQUE:
                return TorqueQueueOptions(**lower_case_options)
            elif queue_system == QueueSystem.LOCAL:
                return LocalQueueOptions(**lower_case_options)
        except pydantic.ValidationError as exception:
            for error in exception.errors():
                _throw_error_or_warning(
                    f"{error['msg']}. Got input '{error['input']}'.",
                    error["input"],
                    is_selected_queue_system,
                )
            return None

    def add_global_queue_options(self, config_dict: ConfigDict) -> None:
        for generic_option in fields(QueueOptions):
            if (
                generic_value := config_dict.get(generic_option.name.upper(), None)  # type: ignore
            ) and self.__dict__[generic_option.name] == generic_option.default:
                try:
                    setattr(self, generic_option.name, generic_value)
                except pydantic.ValidationError as exception:
                    for error in exception.errors():
                        _throw_error_or_warning(
                            f"{error['msg']}. Got input '{error['input']}'.",
                            error["input"],
                            True,
                        )

    @property
    @abstractmethod
    def driver_options(self) -> Dict[str, Any]:
        """Translate the queue options to the key-value API provided by each driver"""


@pydantic.dataclasses.dataclass
class LocalQueueOptions(QueueOptions):
    @property
    def driver_options(self) -> Dict[str, Any]:
        return {}


@pydantic.dataclasses.dataclass
class LsfQueueOptions(QueueOptions):
    bhist_cmd: Optional[NonEmptyString] = None
    bjobs_cmd: Optional[NonEmptyString] = None
    bkill_cmd: Optional[NonEmptyString] = None
    bsub_cmd: Optional[NonEmptyString] = None
    exclude_host: Optional[str] = None
    lsf_queue: Optional[NonEmptyString] = None
    lsf_resource: Optional[str] = None

    @property
    def driver_options(self) -> Dict[str, Any]:
        driver_dict = asdict(self)
        driver_dict["exclude_hosts"] = driver_dict.pop("exclude_host")
        driver_dict["queue_name"] = driver_dict.pop("lsf_queue")
        driver_dict["resource_requirement"] = driver_dict.pop("lsf_resource")
        driver_dict.pop("submit_sleep")
        driver_dict.pop("max_running")
        return driver_dict


@pydantic.dataclasses.dataclass
class TorqueQueueOptions(QueueOptions):
    qsub_cmd: Optional[NonEmptyString] = None
    qstat_cmd: Optional[NonEmptyString] = None
    qdel_cmd: Optional[NonEmptyString] = None
    queue: Optional[NonEmptyString] = None
    memory_per_job: Optional[NonEmptyString] = None
    num_cpus_per_node: pydantic.PositiveInt = 1
    num_nodes: pydantic.PositiveInt = 1
    cluster_label: Optional[NonEmptyString] = None
    job_prefix: Optional[NonEmptyString] = None
    keep_qsub_output: bool = False

    qstat_options: Optional[str] = pydantic.Field(default=None, deprecated=True)
    queue_query_timeout: Optional[str] = pydantic.Field(default=None, deprecated=True)

    @property
    def driver_options(self) -> Dict[str, Any]:
        driver_dict = asdict(self)
        driver_dict["queue_name"] = driver_dict.pop("queue")
        driver_dict.pop("max_running")
        driver_dict.pop("submit_sleep")
        driver_dict.pop("qstat_options")
        driver_dict.pop("queue_query_timeout")
        return driver_dict

    @pydantic.field_validator("memory_per_job")
    @classmethod
    def check_memory_per_job(cls, value: str) -> str:
        if not queue_memory_usage_formats[QueueSystem.TORQUE].validate(value):
            raise ValueError("wrong memory format")
        return value


@pydantic.dataclasses.dataclass
class SlurmQueueOptions(QueueOptions):
    sbatch: NonEmptyString = "sbatch"
    scancel: NonEmptyString = "scancel"
    scontrol: NonEmptyString = "scontrol"
    squeue: NonEmptyString = "squeue"
    exclude_host: str = ""
    include_host: str = ""
    memory: str = ""
    memory_per_cpu: Optional[NonEmptyString] = None
    partition: Optional[NonEmptyString] = None  # aka queue_name
    squeue_timeout: pydantic.PositiveFloat = 2
    max_runtime: Optional[pydantic.NonNegativeFloat] = None

    @property
    def driver_options(self) -> Dict[str, Any]:
        driver_dict = asdict(self)
        driver_dict["sbatch_cmd"] = driver_dict.pop("sbatch")
        driver_dict["scancel_cmd"] = driver_dict.pop("scancel")
        driver_dict["scontrol_cmd"] = driver_dict.pop("scontrol")
        driver_dict["squeue_cmd"] = driver_dict.pop("squeue")
        driver_dict["exclude_hosts"] = driver_dict.pop("exclude_host")
        driver_dict["include_hosts"] = driver_dict.pop("include_host")
        driver_dict["queue_name"] = driver_dict.pop("partition")
        driver_dict.pop("max_running")
        driver_dict.pop("submit_sleep")
        return driver_dict

    @pydantic.field_validator("memory", "memory_per_cpu")
    @classmethod
    def check_memory_per_job(cls, value: str) -> str:
        if not queue_memory_usage_formats[QueueSystem.SLURM].validate(value):
            raise ValueError("wrong memory format")
        return value


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
    QueueSystem.SLURM: QueueMemoryStringFormat(suffixes=["", "K", "M", "G", "T"]),
    QueueSystem.TORQUE: QueueMemoryStringFormat(
        suffixes=["kb", "mb", "gb", "KB", "MB", "GB"]
    ),
}
valid_options: Dict[str, List[str]] = {
    QueueSystem.LOCAL.name: [field.name.upper() for field in fields(LocalQueueOptions)],
    QueueSystem.LSF.name: [field.name.upper() for field in fields(LsfQueueOptions)],
    QueueSystem.SLURM.name: [field.name.upper() for field in fields(SlurmQueueOptions)],
    QueueSystem.TORQUE.name: [
        field.name.upper() for field in fields(TorqueQueueOptions)
    ],
    QueueSystemWithGeneric.GENERIC.name: [
        field.name.upper() for field in fields(QueueOptions)
    ],
}


def _log_duplicated_queue_options(queue_config_list: List[List[str]]) -> None:
    processed_options: Dict[str, str] = {}
    for queue_system, option_name, *values in queue_config_list:
        value = values[0] if values else ""
        if (
            option_name in processed_options
            and processed_options.get(option_name) != value
        ):
            logger.info(
                f"Overwriting QUEUE_OPTION {queue_system} {option_name}:"
                f" \n Old value: {processed_options[option_name]} \n New value: {value}"
            )
        processed_options[option_name] = value


def _raise_for_defaulted_invalid_options(queue_config_list: List[List[str]]) -> None:
    # Invalid options names with no values (i.e. defaulted) are not passed to
    # the validation system, thus we neeed to catch them expliclitly
    for queue_system, option_name, *_ in queue_config_list:
        if option_name not in valid_options[queue_system]:
            raise ConfigValidationError(
                f"Invalid QUEUE_OPTION for {queue_system}: '{option_name}'. "
                f"Valid choices are {sorted(valid_options[queue_system])}."
            )


def _group_queue_options_by_queue_system(
    queue_config_list: List[List[str]],
) -> Dict[QueueSystemWithGeneric, Dict[str, str]]:
    grouped: Dict[QueueSystemWithGeneric, Dict[str, str]] = {}
    for system in QueueSystemWithGeneric:
        grouped[system] = {
            option_line[1]: option_line[2]
            for option_line in queue_config_list
            if option_line[0] in (QueueSystemWithGeneric.GENERIC, system)
            # Empty option values are ignored, yields defaults:
            and len(option_line) > 2
        }
    return grouped


@dataclass
class QueueConfig:
    job_script: str = shutil.which("job_dispatch.py") or "job_dispatch.py"
    realization_memory: int = 0
    max_submit: int = 1
    queue_system: QueueSystem = QueueSystem.LOCAL
    queue_options: QueueOptions = field(default_factory=QueueOptions)
    queue_options_test_run: QueueOptions = field(default_factory=LocalQueueOptions)
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
        realization_memory: int = _parse_realization_memory_str(
            config_dict.get(ConfigKeys.REALIZATION_MEMORY, "0b")
        )
        max_submit: int = config_dict.get(ConfigKeys.MAX_SUBMIT, 1)
        stop_long_running = config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False)

        _raw_queue_options = config_dict.get("QUEUE_OPTION", [])
        _grouped_queue_options = _group_queue_options_by_queue_system(
            _raw_queue_options
        )

        _log_duplicated_queue_options(_raw_queue_options)
        _raise_for_defaulted_invalid_options(_raw_queue_options)

        _all_validated_queue_options = {
            selected_queue_system: QueueOptions.create_queue_options(
                selected_queue_system,
                _grouped_queue_options[selected_queue_system],
                True,
            )
        }
        _all_validated_queue_options.update(
            {
                _queue_system: QueueOptions.create_queue_options(
                    _queue_system, _grouped_queue_options[_queue_system], False
                )
                for _queue_system in QueueSystem
                if _queue_system != selected_queue_system
            }
        )

        queue_options = _all_validated_queue_options[selected_queue_system]
        queue_options_test_run = _all_validated_queue_options[QueueSystem.LOCAL]
        queue_options.add_global_queue_options(config_dict)

        if queue_options.project_code is None:
            tags = {
                fm_name.lower()
                for fm_name, *_ in config_dict.get(ConfigKeys.FORWARD_MODEL, [])
                if fm_name in ["RMS", "FLOW", "ECLIPSE100", "ECLIPSE300"]
            }
            if tags:
                queue_options.project_code = "+".join(tags)

        if selected_queue_system == QueueSystem.TORQUE:
            _check_num_cpu_requirement(
                config_dict.get("NUM_CPU", 1), queue_options, _raw_queue_options
            )

        for _queue_vals in _all_validated_queue_options.values():
            if (
                isinstance(_queue_vals, TorqueQueueOptions)
                and _queue_vals.memory_per_job
                and realization_memory
            ):
                _throw_error_or_warning(
                    "Do not specify both REALIZATION_MEMORY and TORQUE option MEMORY_PER_JOB",
                    "MEMORY_PER_JOB",
                    selected_queue_system == QueueSystem.TORQUE,
                )
            if isinstance(_queue_vals, SlurmQueueOptions) and realization_memory:
                if _queue_vals.memory:
                    _throw_error_or_warning(
                        "Do not specify both REALIZATION_MEMORY and SLURM option MEMORY",
                        "MEMORY",
                        selected_queue_system == QueueSystem.SLURM,
                    )
                if _queue_vals.memory_per_cpu:
                    _throw_error_or_warning(
                        "Do not specify both REALIZATION_MEMORY and SLURM option MEMORY_PER_CPU",
                        "MEMORY_PER_CPU",
                        selected_queue_system == QueueSystem.SLURM,
                    )

        return QueueConfig(
            job_script,
            realization_memory,
            max_submit,
            selected_queue_system,
            queue_options,
            queue_options_test_run,
            stop_long_running=stop_long_running,
        )

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            self.job_script,
            self.realization_memory,
            self.max_submit,
            QueueSystem.LOCAL,
            self.queue_options_test_run,
            self.queue_options_test_run,
            stop_long_running=self.stop_long_running,
        )

    @property
    def max_running(self) -> int:
        return self.queue_options.max_running

    @property
    def submit_sleep(self) -> float:
        return self.queue_options.submit_sleep


def _check_num_cpu_requirement(
    num_cpu: int, torque_options: TorqueQueueOptions, raw_queue_options: List[List[str]]
) -> None:
    flattened_raw_options = [item for line in raw_queue_options for item in line]
    if (
        "NUM_NODES" not in flattened_raw_options
        and "NUM_CPUS_PER_NODE" not in flattened_raw_options
    ):
        return
    if num_cpu != torque_options.num_nodes * torque_options.num_cpus_per_node:
        raise ConfigValidationError(
            f"When NUM_CPU is {num_cpu}, then the product of NUM_NODES ({torque_options.num_nodes}) "
            f"and NUM_CPUS_PER_NODE ({torque_options.num_cpus_per_node}) must be equal."
        )


def _parse_realization_memory_str(realization_memory_str: str) -> int:
    if "-" in realization_memory_str:
        raise ConfigValidationError(
            f"Negative memory does not make sense in {realization_memory_str}"
        )

    if realization_memory_str.isdigit():
        return int(realization_memory_str)
    multipliers = {
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
        "p": 1024**5,
    }
    match = re.search(r"(\d+)\s*(\w)", realization_memory_str)
    if match is None or match.group(2).lower() not in multipliers:
        raise ConfigValidationError(
            f"Could not understand byte unit in {realization_memory_str} {match}"
        )
    return int(match.group(1)) * multipliers[match.group(2).lower()]


def _throw_error_or_warning(
    error_msg: str, option_value: MaybeWithContext, throw_error: bool
) -> None:
    if throw_error:
        raise ConfigValidationError.with_context(
            error_msg,
            option_value,
        ) from None
    else:
        ConfigWarning.warn(
            error_msg,
            option_value,
        )
