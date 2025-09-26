from textwrap import dedent

from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    pre_simulation: list[str] = Field(
        default_factory=list,
        description=dedent(
            """
            The list of workflow jobs triggered pre-simulation.

            Each entry consists of the name of the job, followed by zero or more
            job options.
            """
        ),
    )
    post_simulation: list[str] = Field(
        default_factory=list,
        description=dedent(
            """
            The list of workflow jobs triggered post-simulation.

            Each entry consists of the name of the job, followed by zero or more
            job options.
            """
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
    )
