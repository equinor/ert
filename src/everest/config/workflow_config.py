from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class WorkflowConfig(BaseModel):
    pre_simulation: list[str] | SkipJsonSchema[None] = Field(
        default=None,
        description="List of workflow jobs triggered pre-simulation",
    )
    post_simulation: list[str] | SkipJsonSchema[None] = Field(
        default=None,
        description="List of workflow jobs triggered post-simulation",
    )

    model_config = ConfigDict(
        extra="forbid",
    )
