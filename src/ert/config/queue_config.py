from __future__ import annotations

import logging
import os
import re
import shutil
from abc import abstractmethod
from typing import Annotated, Any, Literal, no_type_check

import pydantic
from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass
from pydantic_core.core_schema import ValidationInfo

from ._get_num_cpu import get_num_cpu_from_data_file
from .parsing import (
    BaseModelWithContextSupport,
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


def activate_script() -> str:
    if venv := os.environ.get("VIRTUAL_ENV"):
        return f"source {venv}/bin/activate"
    if conda_env := os.environ.get("CONDA_ENV"):
        return f'eval "$(conda shell.bash hook)" && conda activate {conda_env}'
    return ""


class QueueOptions(
    BaseModelWithContextSupport,
    validate_assignment=True,
    extra="forbid",
    use_enum_values=True,
    validate_default=True,
):
    name: QueueSystem
    max_running: pydantic.NonNegativeInt = 0
    submit_sleep: pydantic.NonNegativeFloat = 0.0
    num_cpu: pydantic.NonNegativeInt = 1
    realization_memory: pydantic.NonNegativeInt = 0
    job_script: str = shutil.which("fm_dispatch.py") or "fm_dispatch.py"
    project_code: str | None = None
    activate_script: str | None = Field(default=None, validate_default=True)

    @field_validator("activate_script", mode="before")
    @classmethod
    def inject_site_config_script(cls, v: str, info: ValidationInfo) -> str:
        # User value gets highest priority
        if isinstance(v, str):
            return v
        # Use from plugin system if user has not specified
        plugin_script = None
        if info.context:
            plugin_script = info.context.get("activate_script")
        return plugin_script or activate_script()  # Return default value

    @staticmethod
    def create_queue_options(
        queue_system: QueueSystem,
        options: dict[str, Any],
        is_selected_queue_system: bool,
    ) -> QueueOptions | None:
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
        for name, generic_option in QueueOptions.model_fields.items():
            if (generic_value := config_dict.get(name.upper(), None)) and self.__dict__[
                name
            ] == generic_option.default:
                if name == "realization_memory" and isinstance(generic_value, str):
                    generic_value = parse_realization_memory_str(generic_value)
                try:
                    setattr(self, name, generic_value)
                except pydantic.ValidationError as exception:
                    for error in exception.errors():
                        _throw_error_or_warning(
                            f"{error['msg']}. Got input '{error['input']}'.",
                            error["input"],
                            True,
                        )

    @property
    @abstractmethod
    def driver_options(self) -> dict[str, Any]:
        """Translate the queue options to the key-value API provided by each driver"""


class LocalQueueOptions(QueueOptions):
    name: Literal[QueueSystem.LOCAL] = QueueSystem.LOCAL

    @property
    def driver_options(self) -> dict[str, Any]:
        return {}


class LsfQueueOptions(QueueOptions):
    name: Literal[QueueSystem.LSF] = QueueSystem.LSF
    bhist_cmd: NonEmptyString | None = None
    bjobs_cmd: NonEmptyString | None = None
    bkill_cmd: NonEmptyString | None = None
    bsub_cmd: NonEmptyString | None = None
    exclude_host: str | None = None
    lsf_queue: NonEmptyString | None = None
    lsf_resource: str | None = None

    @property
    def driver_options(self) -> dict[str, Any]:
        driver_dict = self.model_dump(
            exclude={
                "name",
                "submit_sleep",
                "max_running",
                "num_cpu",
                "realization_memory",
                "job_script",
            }
        )
        driver_dict["exclude_hosts"] = driver_dict.pop("exclude_host")
        driver_dict["queue_name"] = driver_dict.pop("lsf_queue")
        driver_dict["resource_requirement"] = driver_dict.pop("lsf_resource")
        return driver_dict


class TorqueQueueOptions(QueueOptions):
    name: Literal[QueueSystem.TORQUE] = QueueSystem.TORQUE
    qsub_cmd: NonEmptyString | None = None
    qstat_cmd: NonEmptyString | None = None
    qdel_cmd: NonEmptyString | None = None
    queue: NonEmptyString | None = None
    cluster_label: NonEmptyString | None = None
    job_prefix: NonEmptyString | None = None
    keep_qsub_output: bool = False

    @property
    def driver_options(self) -> dict[str, Any]:
        driver_dict = self.model_dump(
            exclude={
                "name",
                "max_running",
                "submit_sleep",
                "num_cpu",
                "realization_memory",
                "job_script",
            }
        )
        driver_dict["queue_name"] = driver_dict.pop("queue")
        return driver_dict


class SlurmQueueOptions(QueueOptions):
    name: Literal[QueueSystem.SLURM] = QueueSystem.SLURM
    sbatch: NonEmptyString = "sbatch"
    scancel: NonEmptyString = "scancel"
    scontrol: NonEmptyString = "scontrol"
    sacct: NonEmptyString = "sacct"
    squeue: NonEmptyString = "squeue"
    exclude_host: str = ""
    include_host: str = ""
    partition: NonEmptyString | None = None  # aka queue_name
    squeue_timeout: pydantic.PositiveFloat = 2
    max_runtime: pydantic.NonNegativeFloat | None = None

    @property
    def driver_options(self) -> dict[str, Any]:
        driver_dict = self.model_dump(
            exclude={
                "name",
                "max_running",
                "submit_sleep",
                "num_cpu",
                "realization_memory",
                "job_script",
            }
        )
        driver_dict["sbatch_cmd"] = driver_dict.pop("sbatch")
        driver_dict["scancel_cmd"] = driver_dict.pop("scancel")
        driver_dict["scontrol_cmd"] = driver_dict.pop("scontrol")
        driver_dict["sacct_cmd"] = driver_dict.pop("sacct")
        driver_dict["squeue_cmd"] = driver_dict.pop("squeue")
        driver_dict["exclude_hosts"] = driver_dict.pop("exclude_host")
        driver_dict["include_hosts"] = driver_dict.pop("include_host")
        driver_dict["queue_name"] = driver_dict.pop("partition")
        return driver_dict


@dataclass
class QueueMemoryStringFormat:
    suffixes: list[str]

    def validate(self, mem_str_format: str | None) -> bool:
        if mem_str_format is None:
            return True
        return (
            re.match(
                r"\d+(" + "|".join(self.suffixes) + ")$",
                mem_str_format,
            )
            is not None
        )


torque_memory_usage_format: QueueMemoryStringFormat = QueueMemoryStringFormat(
    suffixes=["kb", "mb", "gb", "KB", "MB", "GB"]
)

valid_options: dict[str, list[str]] = {
    QueueSystem.LOCAL: [field.upper() for field in LocalQueueOptions.model_fields],
    QueueSystem.LSF: [field.upper() for field in LsfQueueOptions.model_fields],
    QueueSystem.SLURM: [field.upper() for field in SlurmQueueOptions.model_fields],
    QueueSystem.TORQUE: [field.upper() for field in TorqueQueueOptions.model_fields],
    QueueSystemWithGeneric.GENERIC: [
        field.upper() for field in QueueOptions.model_fields
    ],
}


def _log_duplicated_queue_options(queue_config_list: list[list[str]]) -> None:
    processed_options: dict[str, str] = {}
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


def _raise_for_defaulted_invalid_options(queue_config_list: list[list[str]]) -> None:
    # Invalid options names with no values (i.e. defaulted) are not passed to
    # the validation system, thus we neeed to catch them expliclitly
    for queue_system, option_name, *_ in queue_config_list:
        if option_name not in valid_options[queue_system]:
            raise ConfigValidationError.with_context(
                f"Invalid QUEUE_OPTION for {queue_system}: '{option_name}'. "
                f"Valid choices are {sorted(valid_options[queue_system])}.",
                option_name,
            )


def _group_queue_options_by_queue_system(
    queue_config_list: list[list[str]],
) -> dict[QueueSystemWithGeneric, dict[str, str]]:
    grouped: dict[QueueSystemWithGeneric, dict[str, str]] = {}
    for system in QueueSystemWithGeneric:
        grouped[system] = {
            option_line[1]: option_line[2]
            for option_line in queue_config_list
            if option_line[0] in {QueueSystemWithGeneric.GENERIC, system}
            # Empty option values are ignored, yields defaults:
            and len(option_line) > 2
        }
    return grouped


class QueueConfig(BaseModel):
    max_submit: int = 1
    queue_system: QueueSystem = QueueSystem.LOCAL
    queue_options: (
        LsfQueueOptions | TorqueQueueOptions | SlurmQueueOptions | LocalQueueOptions
    ) = pydantic.Field(default_factory=LocalQueueOptions, discriminator="name")
    stop_long_running: bool = False
    max_runtime: int | None = None

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> QueueConfig:
        selected_queue_system = QueueSystem(
            config_dict.get("QUEUE_SYSTEM", QueueSystem.LOCAL)
        )
        job_script: str = config_dict.get(
            "JOB_SCRIPT", shutil.which("fm_dispatch.py") or "fm_dispatch.py"
        )
        config_dict["JOB_SCRIPT"] = job_script
        max_submit: int = config_dict.get(ConfigKeys.MAX_SUBMIT, 1)
        stop_long_running = config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False)

        if (
            ConfigKeys.NUM_CPU not in config_dict
            and ConfigKeys.DATA_FILE in config_dict
        ):
            data_file = config_dict.get(ConfigKeys.DATA_FILE)
            if num_cpu := get_num_cpu_from_data_file(data_file):
                logger.info(f"Parsed NUM_CPU={num_cpu} from {data_file}")
                config_dict[ConfigKeys.NUM_CPU] = num_cpu

        raw_queue_options = config_dict.get("QUEUE_OPTION", [])
        grouped_queue_options = _group_queue_options_by_queue_system(raw_queue_options)
        _log_duplicated_queue_options(raw_queue_options)
        _raise_for_defaulted_invalid_options(raw_queue_options)

        all_validated_queue_options = {
            selected_queue_system: QueueOptions.create_queue_options(
                selected_queue_system,
                grouped_queue_options[selected_queue_system],
                True,
            )
        }
        all_validated_queue_options.update(
            {
                _queue_system: QueueOptions.create_queue_options(
                    _queue_system, grouped_queue_options[_queue_system], False
                )
                for _queue_system in QueueSystem
                if _queue_system != selected_queue_system
            }
        )

        queue_options = all_validated_queue_options[selected_queue_system]
        queue_options.add_global_queue_options(config_dict)

        if queue_options.project_code is None:
            tags = {
                fm_name.lower()
                for fm_name, *_ in config_dict.get(ConfigKeys.FORWARD_MODEL, [])
                if fm_name in {"RMS", "FLOW", "ECLIPSE100", "ECLIPSE300"}
            }
            if tags:
                queue_options.project_code = "+".join(tags)

        return QueueConfig(
            max_submit=max_submit,
            queue_system=selected_queue_system,
            queue_options=queue_options,
            stop_long_running=bool(stop_long_running),
            max_runtime=config_dict.get(ConfigKeys.MAX_RUNTIME),
        )

    def create_local_copy(self) -> QueueConfig:
        return QueueConfig(
            max_submit=self.max_submit,
            queue_system=QueueSystem.LOCAL,
            queue_options=LocalQueueOptions(max_running=self.max_running),
            stop_long_running=bool(self.stop_long_running),
            max_runtime=self.max_runtime,
        )

    @property
    def max_running(self) -> int:
        return self.queue_options.max_running

    @property
    def submit_sleep(self) -> float:
        return self.queue_options.submit_sleep


def parse_realization_memory_str(realization_memory_str: str) -> int:
    if "-" in realization_memory_str:
        raise ConfigValidationError.with_context(
            f"Negative memory does not make sense in {realization_memory_str}",
            realization_memory_str,
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
        raise ConfigValidationError.with_context(
            f"Could not understand byte unit in {realization_memory_str}",
            realization_memory_str,
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
