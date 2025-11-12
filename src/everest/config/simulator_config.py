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
            Amount of memory to set aside for a forward model.

            This information is propagated to the queue system as the amount of memory
            to reserve/book for a realization to complete. It is up to the configuration
            of the queuing system how to treat this information, but usually it will
            stop more realizations being assigned to a compute node if the compute nodes
            memory is already fully booked.

            Setting this number lower than the peak memory consumption of each
            realization puts the realization at risk of being killed in an out-of-memory
            situation. Setting this number higher than needed will give longer wait
            times in the queue.

            For the local queue system, this keyword has no effect. In that scenario,
            you can use `max_running`  to choke the memory consumption.
            scheduling of compute jobs.

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

    @model_validator(mode="before")
    @classmethod
    def apply_site_or_default_queue_if_no_user_queue(
        cls, data: dict[str, Any], info: ValidationInfo
    ) -> Any:
        queue_system = data.get("queue_system")
        if queue_system is None:
            options = None
            if info.context:
                options = info.context.queue_options

            defaulted_queue_options = (
                options.model_dump()
                if options is not None
                else LocalQueueOptions(max_running=8).model_dump()
            )

            user_configured_max_memory = data.get("max_memory")
            if user_configured_max_memory is not None:
                cls.validate_max_memory(max_memory=user_configured_max_memory)
                defaulted_queue_options["realization_memory"] = (
                    user_configured_max_memory
                )

            user_configured_cores_per_node = data.get("cores_per_node")
            if user_configured_cores_per_node is not None:
                defaulted_queue_options["num_cpu"] = user_configured_cores_per_node

            data["queue_system"] = defaulted_queue_options
            data["max_memory"] = None

        return data

    @field_validator("max_memory", mode="before")
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
        if config.max_memory is None:
            return config
        parsed_max_memory = (
            parse_string_to_bytes(config.max_memory)
            if type(config.max_memory) is str
            else int(config.max_memory)
        )
        if (
            config.queue_system is not None
            and config.queue_system.realization_memory == 0
        ):
            config.queue_system.realization_memory = parsed_max_memory
        elif (
            config.queue_system is not None
            and config.queue_system.realization_memory > 0
            and config.queue_system.realization_memory != parsed_max_memory
        ):
            raise ConfigValidationError(
                "Ambiguous configuration of realization_memory. "
                "Specify either max_memory or realization_memory, not both"
            )

        return config
