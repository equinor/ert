from typing import Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from .has_ert_queue_options import HasErtQueueOptions


class SimulatorConfig(BaseModel, HasErtQueueOptions, extra="forbid"):  # type: ignore
    name: Optional[str] = Field(
        default=None, description="Specifies which queue to use"
    )
    cores: Optional[PositiveInt] = Field(
        default=None,
        description="""Defines the number of simultaneously running forward models.

    When using queue system lsf, this corresponds to number of nodes used at one
    time, whereas when using the local queue system, cores refers to the number of
    cores you want to use on your system.

    This number is specified in Ert as MAX_RUNNING.
    """,
    )
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
    exclude_host: Optional[str] = Field(
        None,
        description="""Comma separated list of nodes that should be
                 excluded from the slurm run.""",
    )
    include_host: Optional[str] = Field(
        None,
        description="""Comma separated list of nodes that
                should be included in the slurm run""",
    )
    max_memory: Optional[str] = Field(
        default=None,
        description="Maximum memory usage for a slurm job.",
    )
    max_memory_cpu: Optional[str] = Field(
        default=None,
        description="Maximum memory usage per cpu for a slurm job.",
    )
    max_runtime: Optional[NonNegativeInt] = Field(
        default=None,
        description="""Maximum allowed running time of a forward model. When
        set, a job is only allowed to run for max_runtime seconds.
        A value of 0 means unlimited runtime.
        """,
    )
    options: Optional[str] = Field(
        default=None,
        description="""Used to specify options to LSF.
        Examples to set memory requirement is:
        * rusage[mem=1000]""",
    )
    queue_system: Optional[Literal["lsf", "local", "slurm"]] = Field(
        default="local",
        description="Defines which queue system the everest server runs on.",
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
    sbatch: Optional[str] = Field(
        default=None,
        description="sbatch executable to be used by the slurm queue interface.",
    )
    scancel: Optional[str] = Field(
        default=None,
        description="scancel executable to be used by the slurm queue interface.",
    )
    scontrol: Optional[str] = Field(
        default=None,
        description="scontrol executable to be used by the slurm queue interface.",
    )
    squeue: Optional[str] = Field(
        default=None,
        description="squeue executable to be used by the slurm queue interface.",
    )
    server: Optional[str] = Field(default=None, description="Name of LSF server to use")
    slurm_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for cached status used by the slurm queue interface",
    )
    squeue_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for cached status used by the slurm queue interface.",
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
