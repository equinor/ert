from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):  # type: ignore
    pre_simulation: Optional[List[str]] = Field(
        default=None,
        description="List of workflow jobs triggered pre-simulation",
    )
    post_simulation: Optional[List[str]] = Field(
        default=None,
        description="List of workflow jobs triggered post-simulation",
    )

    model_config = ConfigDict(
        extra="forbid",
    )
