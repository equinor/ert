from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .has_ert_queue_options import HasErtQueueOptions


class ServerConfig(BaseModel, HasErtQueueOptions):  # type: ignore
    name: Optional[str] = Field(
        None,
        description="""Specifies which queue to use.

Examples are
* mr
* bigmem

The everest server generally has lower resource requirements than forward models such
as RMS and Eclipse.
    """,
    )  # Corresponds to queue name
    exclude_host: Optional[str] = Field(
        None,
        description="""Comma separated list of nodes that should be
         excluded from the slurm run""",
    )
    include_host: Optional[str] = Field(
        None,
        description="""Comma separated list of nodes that
        should be included in the slurm run""",
    )
    options: Optional[str] = Field(
        None,
        description="""Used to specify options to LSF.
        Examples to set memory requirement is:
        * rusage[mem=1000]""",
    )
    queue_system: Optional[Literal["lsf", "local", "slurm"]] = Field(
        None,
        description="Defines which queue system the everest server runs on.",
    )
    model_config = ConfigDict(
        extra="forbid",
    )
