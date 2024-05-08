from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everest.optimizer.utils import get_ropt_plugin_manager


class SamplerConfig(BaseModel):  # type: ignore
    backend: Optional[str] = Field(
        default=None,
        description="""The backend used by Everest for sampling points.

The sampler backend provides the methods for sampling the points used to
estimate the gradient. The default is the built-in 'scipy' backend.

""",
    )
    backend_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
Specifies a dict of optional parameters for the sampler backend.

This dict of values is passed unchanged to the selected method in the backend.

""",
    )
    method: Optional[str] = Field(
        default=None,
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
        method = "default" if self.method is None else self.method
        backend = "scipy" if self.backend is None else self.backend
        if not get_ropt_plugin_manager().is_supported("sampler", f"{backend}/{method}"):
            raise ValueError(f"Sampler '{backend}/{method}' not found")
        return self

    model_config = ConfigDict(
        extra="forbid",
        metadata={
            "doc": """
A sampler specification section applies to a group of controls, or to an
individual control. Sampler specifications are not required, with the following
behavior, if no sampler sections are provided, a normal distribution is used.

If at least one control group or variable has a sampler specification, only the
groups or variables with a sampler specification are perturbed.
Controls/variables that do not have a sampler section will not be perturbed at
all. If that is not desired, make sure to specify a sampler for each control
group and/or variable (or none at all to use a normal distribution for each
control).

Within the sampler section, the *shared* keyword can be used to direct the sampler
to use the same perturbations for each realization.

""",
        },
    )
