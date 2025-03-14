from typing import Any

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.plugins import ErtPluginManager

simulator_example = {"queue_system": {"name": "local", "max_running": 3}}


def check_removed_config(queue_system: Any) -> None:
    queue_systems = {
        "lsf": LsfQueueOptions,
        "torque": TorqueQueueOptions,
        "slurm": SlurmQueueOptions,
        "local": LocalQueueOptions,
    }
    if isinstance(queue_system, str) and queue_system in queue_systems:
        raise ValueError(
            f"Queue system configuration has changed, valid options for {queue_system} are: {list(queue_systems[queue_system].__dataclass_fields__.keys())}"  # type: ignore
        )


class SimulatorConfig(BaseModel, extra="forbid"):
    cores_per_node: PositiveInt | None = Field(
        default=None,
        description="""defines the number of CPUs when running
     the forward models. This can for example be used in conjunction with the Eclipse
     parallel keyword for multiple CPU simulation runs. This keyword has no effect
     when running with the local queue.

    This number is specified in Ert as NUM_CPU.""",
    )
    delete_run_path: bool | None = Field(
        default=None,
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
    resubmit_limit: NonNegativeInt | None = Field(
        default=None,
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
    def default_local_queue(cls, v: Any) -> Any:
        if v is None:
            return LocalQueueOptions(max_running=8)
        if "activate_script" not in v and (
            active_script := ErtPluginManager().activate_script()
        ):
            v["activate_script"] = active_script
        return v

    @model_validator(mode="before")
    @classmethod
    def check_old_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            check_removed_config(data.get("queue_system"))
        return data
