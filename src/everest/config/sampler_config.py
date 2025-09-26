import logging
from textwrap import dedent
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everest.optimizer.utils import get_ropt_plugin_manager
from everest.strings import EVEREST


class SamplerConfig(BaseModel):
    backend: str | None = Field(
        default=None,
        description=dedent(
            """
            [Deprecated]

            The correct backend will be inferred by the method. If several backends
            have a method named `A`, pick a specific backend `B` by putting `B/A` in
            the `method` field.
            """
        ),
    )
    method: str = Field(
        default="norm",
        description=dedent(
            """
            The sampling method or distribution used by the sampler backend.

            The set of available methods depends on the sampler backend used. By
            default a plugin based on `scipy.stats` is used, implementing the
            following methods:

            - From Probability Distributions:
                - `norm`: Samples from a standard normal distribution (mean 0,
                  standard deviation 1).
                - `truncnorm`: Samples from a truncated normal distribution
                  (mean 0, std. dev. 1), truncated to the range `[-1, 1]`.
                - `uniform`: Samples from a uniform distribution in the range
                  `[-1, 1]`.

            - From Quasi-Monte Carlo Sequences:
                - `sobol`: Uses Sobol' sequences.
                - `halton`: Uses Halton sequences.
                - `lhs`: Uses Latin Hypercube Sampling.

                Note: QMC samples are generated in the unit hypercube `[0, 1]^d`
                and then scaled to the hypercube `[-1, 1]^d`.
            """
        ),
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description=dedent(
            """
            Specifies a dict of optional parameters for the sampler backend.

            This dict of values is passed unchanged to the selected method in
            the backend.
            """
        ),
    )
    shared: bool | None = Field(
        default=None,
        description=dedent(
            """
            Whether to share perturbations between realizations.
            """
        ),
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_backend_and_method(self) -> Self:
        if self.backend is not None:
            message = (
                "sampler.backend is deprecated. "
                "The correct backend will be inferred by the method. "
                "If several backends have a method named A, you need to pick "
                "a specific backend B by putting B/A in sampler.method."
            )
            print(message)
            logging.getLogger(EVEREST).warning(message)

        # Update the default for backends that are not scipy:
        if (
            self.backend not in {None, "scipy"}
            and "method" not in self.model_fields_set
        ):
            self.method = "default"

        if self.backend is not None:
            self.method = f"{self.backend}/{self.method}"

        if (
            get_ropt_plugin_manager().get_plugin_name("sampler", f"{self.method}")
            is None
        ):
            raise ValueError(f"Sampler method '{self.method}' not found")

        self.backend = None

        return self
