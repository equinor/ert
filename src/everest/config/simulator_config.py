from typing import Optional, Union

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)


class SimulatorConfig(BaseModel, extra="forbid"):  # type: ignore
    cores_per_node: Optional[PositiveInt] = Field(
        default=None,
        description="""defines the number of CPUs when running
     the forward models. This can for example be used in conjunction with the Eclipse
     parallel keyword for multiple CPU simulation runs. This keyword has no effect
     when running with the local queue.

    This number is specified in Ert as NUM_CPU.""",
    )
    delete_run_path: Optional[bool] = Field(
        default=None,
        description="Whether the batch folder for a successful simulation "
        "needs to be deleted.",
    )
    max_runtime: Optional[NonNegativeInt] = Field(
        default=None,
        description="""Maximum allowed running time of a forward model. When
        set, a job is only allowed to run for max_runtime seconds.
        A value of 0 means unlimited runtime.
        """,
    )
    queue_system: Union[
        LocalQueueOptions, LsfQueueOptions, SlurmQueueOptions, TorqueQueueOptions, None
    ] = Field(
        default=None,
        description="Defines which queue system the everest submits jobs to",
        discriminator="name",
        validate_default=True,
    )
    resubmit_limit: Optional[NonNegativeInt] = Field(
        default=None,
        description="""
        Defines how many times should the queue system retry a forward model.

    A forward model may fail for reasons that are not due to the forward model
    itself, like a node in the cluster crashing, network issues, etc.  Therefore, it
    might make sense to resubmit a forward model in case it fails.
    resumbit_limit defines the number of times we will resubmit a failing forward model.
    If not specified, a default value of 1 will be used.""",
    )
    enable_cache: bool = Field(
        default=False,
        description="""Enable forward model result caching.

        If enabled, objective and constraint function results are cached for
        each realization. If the optimizer requests an evaluation that has
        already been done before, these cached values will be re-used without
        running the forward model again.

        This option is disabled by default, since it will not be necessary for
        the most common use of a standard optimization with a continuous
        optimizer.""",
    )

    @field_validator("queue_system", mode="before")
    @classmethod
    def default_local_queue(cls, v):
        if v is None:
            return LocalQueueOptions(max_running=8)
        return v
