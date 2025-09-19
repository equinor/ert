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
