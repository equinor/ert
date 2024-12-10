import warnings
from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator

from .has_ert_queue_options import HasErtQueueOptions


class SimulatorConfig(BaseModel, HasErtQueueOptions, extra="forbid"):  # type: ignore
    name: str | None = Field(default=None, description="Specifies which queue to use")
    cores: PositiveInt | None = Field(
        default=None,
        description="""Defines the number of simultaneously running forward models.

    When using queue system lsf, this corresponds to number of nodes used at one
    time, whereas when using the local queue system, cores refers to the number of
    cores you want to use on your system.

    This number is specified in Ert as MAX_RUNNING.
    """,
    )
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
    exclude_host: str | None = Field(
        "",
        description="""Comma separated list of nodes that should be
                 excluded from the slurm run.""",
    )
    include_host: str | None = Field(
        "",
        description="""Comma separated list of nodes that
                should be included in the slurm run""",
    )
    max_runtime: NonNegativeInt | None = Field(
        default=None,
        description="""Maximum allowed running time of a forward model. When
        set, a job is only allowed to run for max_runtime seconds.
        A value of 0 means unlimited runtime.
        """,
    )
    options: str | None = Field(
        default=None,
        description="""Used to specify options to LSF.
        Examples to set memory requirement is:
        * rusage[mem=1000]""",
    )
    queue_system: Literal["lsf", "local", "slurm", "torque"] | None = Field(
        default="local",
        description="Defines which queue system the everest server runs on.",
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
    sbatch: str | None = Field(
        default=None,
        description="sbatch executable to be used by the slurm queue interface.",
    )
    scancel: str | None = Field(
        default=None,
        description="scancel executable to be used by the slurm queue interface.",
    )
    scontrol: str | None = Field(
        default=None,
        description="scontrol executable to be used by the slurm queue interface.",
    )
    sacct: str | None = Field(
        default=None,
        description="sacct executable to be used by the slurm queue interface.",
    )
    squeue: str | None = Field(
        default=None,
        description="squeue executable to be used by the slurm queue interface.",
    )
    server: str | None = Field(
        default=None,
        description="Name of LSF server to use. This option is deprecated and no longer required",
    )
    slurm_timeout: int | None = Field(
        default=None,
        description="Timeout for cached status used by the slurm queue interface",
    )
    squeue_timeout: int | None = Field(
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
    qsub_cmd: str | None = Field(default="qsub", description="The submit command")
    qstat_cmd: str | None = Field(default="qstat", description="The query command")
    qdel_cmd: str | None = Field(default="qdel", description="The kill command")
    cluster_label: str | None = Field(
        default=None,
        description="The name of the cluster you are running simulations in.",
    )
    keep_qsub_output: int | None = Field(
        default=0,
        description="Set to 1 to keep error messages from qsub. Usually only to be used if somethign is seriously wrong with the queue environment/setup.",
    )
    submit_sleep: float | None = Field(
        default=0.5,
        description="To avoid stressing the TORQUE/PBS system you can instruct the driver to sleep for every submit request. The argument to the SUBMIT_SLEEP is the number of seconds to sleep for every submit, which can be a fraction like 0.5",
    )
    project_code: str | None = Field(
        default=None,
        description="String identifier used to map hardware resource usage to a project or account. The project or account does not have to exist.",
    )

    @field_validator("server")
    @classmethod
    def validate_server(cls, server):  # pylint: disable=E0213
        if server is not None and server:
            warnings.warn(
                "The simulator server property was deprecated and is no longer needed",
                DeprecationWarning,
                stacklevel=1,
            )
