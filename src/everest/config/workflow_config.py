from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    pre_simulation: list[str] = Field(
        default_factory=list,
        description="List of workflow jobs triggered pre-simulation",
    )
    post_simulation: list[str] = Field(
        default_factory=list,
        description="List of workflow jobs triggered post-simulation",
    )

    model_config = ConfigDict(
        extra="forbid",
    )
