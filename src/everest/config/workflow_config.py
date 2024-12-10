from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):  # type: ignore
    pre_simulation: list[str] | None = Field(
        default=None,
        description="List of workflow jobs triggered pre-simulation",
    )
    post_simulation: list[str] | None = Field(
        default=None,
        description="List of workflow jobs triggered post-simulation",
    )

    model_config = ConfigDict(
        extra="forbid",
    )
