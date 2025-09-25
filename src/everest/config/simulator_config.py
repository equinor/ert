from textwrap import dedent
from typing import Any

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo

from ert.base_model_context import BaseModelWithContextSupport
from ert.config import ConfigValidationError
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
    parse_string_to_bytes,
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
        description=dedent(
            """
            Defines the number of CPUs when running the forward models. This can
            for example be used in conjunction with the Eclipse parallel keyword
            for multiple CPU simulation runs.

            This keyword has no effect when running with the local queue.
            """
        ),
    )
    delete_run_path: bool = Field(
        default=False,
        description=dedent(
            """
            If `True`, delete the output folder of a successful forward model run.
            """
        ),
    )
    max_runtime: NonNegativeInt | None = Field(
        default=None,
        description=dedent(
            """
            Maximum allowed running time for each individual forward model job
            specified in seconds.

            A value of 0 means unlimited runtime.
            """
        ),
    )
    max_memory: int | str | None = Field(
        default=None,
        description=dedent(
            """
            Maximum allowed memory usage of a forward model.

            When set, a job is only allowed to use `max_memory` of memory.

            `max_memory` may be an integer value, indicating the number of
            bytes, or a string consisting of a number followed by a unit. The
            unit indicates the multiplier that is applied, and must start with
            one of these characters:

            - b, B: bytes
            - k, K: kilobytes (1024 bytes)
            - m, M: megabytes (1024**2 bytes)
            - g, G: gigabytes (1024**3 bytes)
            - t, T: terabytes (1024**4 bytes)
            - p, P: petabytes (1024**5 bytes)

            Spaces between the number and the unit are ignored, and so are any
            characters after the first. For example: 2g, 2G, and 2 GB all
            resolve to the same value: 2 gigabytes, equaling 2 * 1024**3 bytes.

            If not set, or a set to zero, the allowed amount of memory is
            unlimited.
            """
        ),
    )
    queue_system: (
        LocalQueueOptions
        | LsfQueueOptions
        | SlurmQueueOptions
        | TorqueQueueOptions
        | None
    ) = Field(
        default=None,
        description=dedent(
            """
            Defines which queue system the everest submits jobs to.
            """
        ),
        discriminator="name",
        validate_default=True,
    )
    resubmit_limit: NonNegativeInt = Field(
        default=1,
        description=dedent(
            """
        Specifies how many times a forward model may be submitted to the queue
        system.

        A forward model may fail for external reasons, examples include node in
        the cluster crashing, network issues, etc. Therefore, it might make
        sense to resubmit a forward model in case it fails. `resubmit_limit`
        defines the number of times we will resubmit a failing forward model. If
        not specified, a default value of 1 will be used.
        """
        ),
    )

    @field_validator("queue_system", mode="before")
    @classmethod
    def default_local_queue(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            options = None
            if info.context:
                options = info.context.queue_options
            return options or LocalQueueOptions(max_running=8)
        return v

    @field_validator("max_memory")
    @classmethod
    def validate_max_memory(cls, max_memory: int | str | None) -> str | None:
        if max_memory is None:
            return None
        max_memory = str(max_memory).strip()
        try:
            parse_string_to_bytes(max_memory)
        except ConfigValidationError as exc:
            raise ValueError(exc.cli_message) from exc
        return max_memory

    @model_validator(mode="before")
    @classmethod
    def check_old_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            check_removed_config(data.get("queue_system"))
        return data

    @model_validator(mode="after")
    def update_max_memory(config: "SimulatorConfig") -> "SimulatorConfig":
        if (
            config.max_memory is not None
            and config.queue_system is not None
            and config.queue_system.realization_memory == 0
        ):
            config.queue_system.realization_memory = (
                parse_string_to_bytes(config.max_memory)
                if type(config.max_memory) is str
                else int(config.max_memory)
            )
        return config
