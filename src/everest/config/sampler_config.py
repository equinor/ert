from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everest.optimizer.utils import get_ropt_plugin_manager


class SamplerConfig(BaseModel):  # type: ignore
    backend: str = Field(
        default="scipy",
        description="""The backend used by Everest for sampling points.

The sampler backend provides the methods for sampling the points used to
estimate the gradient. The default is the built-in 'scipy' backend.

""",
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="backend_options",
        description="""
Specifies a dict of optional parameters for the sampler backend.

This dict of values is passed unchanged to the selected method in the backend.

""",
    )
    method: str = Field(
        default="default",
        description="""The sampling method or distribution used by the sampler backend.
""",
    )
    shared: Optional[bool] = Field(
        default=None,
        description="""Whether to share perturbations between realizations.
""",
    )

    @model_validator(mode="after")
    def validate_backend_and_method(self):  # pylint: disable=E0213
        if not get_ropt_plugin_manager().is_supported("sampler", f"{self.ropt_method}"):
            raise ValueError(f"Sampler '{self.backend}/{self.method}' not found")
        return self

    @property
    def ropt_method(self) -> str:
        return f"{self.backend}/{self.method}"

    model_config = ConfigDict(
        extra="forbid",
    )
