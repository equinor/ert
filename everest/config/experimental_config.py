from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ExperimentalConfig(BaseModel, extra="forbid"):  # type: ignore
    plan: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""Optional optimization plan.""",
    )
