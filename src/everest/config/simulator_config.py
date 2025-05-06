from typing import Any

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo

from ert.config import ConfigValidationError
from ert.config.parsing import BaseModelWithContextSupport
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    parse_realization_memory_str,
)

simulator_example = {"queue_system": {"name": "local", "max_running": 3}}


def check_removed_config(queue_system: Any) -> None:
    queue_systems: dict[str, type[QueueOptions]] = {
        "lsf": LsfQueueOptions,
        "torque": TorqueQueueOptions,
        "slurm": SlurmQueueOptions,
        "local": LocalQueueOptions,
    }
    if isinstance(queue_system, str) and queue_system in queue_systems:
        raise ValueError(
            "Queue system configuration has changed, valid options "
            f"for {queue_system} "
            f"are: {list(queue_systems[queue_system].model_fields.keys())}"
        )


class SimulatorConfig(BaseModelWithContextSupport, extra="forbid"):
    cores_per_node: PositiveInt | None = Field(
        default=None,
        description="""defines the number of CPUs when running
     the forward models. This can for example be used in conjunction with the Eclipse
     parallel keyword for multiple CPU simulation runs. This keyword has no effect
     when running with the local queue.

    This number is specified in Ert as NUM_CPU.""",
    )
    delete_run_path: bool = Field(
        default=False,
        description="Whether the batch folder for a successful simulation "
        "needs to be deleted.",
    )
    max_runtime: NonNegativeInt | None = Field(
        default=None,
        description="""Maximum allowed running time of a forward model. When
        set, a job is only allowed to run for max_runtime seconds.
        A value of 0 means unlimited runtime.
        """,
    )
    max_memory: int | str | None = Field(
        default=None,
        description="""Maximum allowed memory usage of a forward model. When
        set, a job is only allowed to use max_memory of memory.

        max_memory may be an integer value, indicating the number of bytes, or a
        string consisting of a number followed by a unit. The unit indicates the
        multiplier that is applied, and must start with one of these characters:

        * b, B: bytes
        * k, K: kilobytes (1024 bytes)
        * m, M: megabytes (1024**2 bytes)
        * g, G: gigabytes (1024**3 bytes)
        * t, T: terabytes (1024**4 bytes)
        * p, P: petabytes (1024**5 bytes)

        Spaces between the number and the unit are ignored, and so are any
        characters after the first. For example: 2g, 2G, and 2 GB all resolve
        to the same value: 2 gigabytes, equaling 2 * 1024**3 bytes.

        If not set, or a set to zero, the allowed amount of memory is unlimited.
        """,
    )
    queue_system: (
        LocalQueueOptions
        | LsfQueueOptions
        | SlurmQueueOptions
        | TorqueQueueOptions
        | None
    ) = Field(
        default=None,
        description="Defines which queue system the everest submits jobs to",
        discriminator="name",
        validate_default=True,
    )
    resubmit_limit: NonNegativeInt = Field(
        default=1,
        description="""
        Defines how many times should the queue system retry a forward model.

    A forward model may fail for reasons that are not due to the forward model
    itself, like a node in the cluster crashing, network issues, etc.  Therefore, it
    might make sense to resubmit a forward model in case it fails.
    resumbit_limit defines the number of times we will resubmit a failing forward model.
    If not specified, a default value of 1 will be used.""",
    )

    @field_validator("queue_system", mode="before")
    @classmethod
    def default_local_queue(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            options = None
            if info.context:
                options = info.context.get("queue_system")
            return options or LocalQueueOptions(max_running=8)
        return v

    @field_validator("max_memory")
    @classmethod
    def validate_max_memory(cls, max_memory: int | str | None) -> str | None:
        if max_memory is None:
            return None
        max_memory = str(max_memory).strip()
        try:
            parse_realization_memory_str(max_memory)
        except ConfigValidationError as exc:
            raise ValueError(exc.cli_message) from exc
        return max_memory

    @model_validator(mode="before")
    @classmethod
    def check_old_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            check_removed_config(data.get("queue_system"))
        return data
