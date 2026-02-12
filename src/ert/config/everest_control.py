from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, Self

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ropt.workflow import find_sampler_plugin

from .parameter_config import ParameterCardinality, ParameterConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

    Number = int | float
    DataType = Mapping[str, Number | Mapping[str, Number]]
    MutableDataType = MutableMapping[str, Number | MutableMapping[str, Number]]


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
            # Note: Importing EVEREST.everest
            # leads to circular import, but we still wish to log
            # from "everest" here as per old behavior.
            # Can consider logging this as if from ERT,
            # which is valid if we store SamplerConfig as part of
            # EverestControl configs.
            logging.getLogger("everest").warning(message)

        # Update the default for backends that are not scipy:
        if (
            self.backend not in {None, "scipy"}
            and "method" not in self.model_fields_set
        ):
            self.method = "default"

        if self.backend is not None:
            self.method = f"{self.backend}/{self.method}"

        try:
            plugin = find_sampler_plugin(f"{self.method}")
        except ValueError:
            raise
        except Exception as exc:
            ert_version = importlib.metadata.version("ert")
            ropt_version = importlib.metadata.version("ropt")
            msg = (
                f"Error while initializing ropt:\n\n{exc}.\n\n"
                "There may a be version mismatch between "
                f"ERT ({ert_version}) and ropt ({ropt_version})\n"
                "If the installation is correct, please report this as a bug."
            )
            raise RuntimeError(msg) from exc

        if plugin is None:
            raise ValueError(f"Sampler method '{self.method}' not found")

        self.backend = None

        return self


class EverestControl(ParameterConfig):
    """Create an EverestControl for a single control variable.

    Each EverestControl represents one scalar value. Multiple controls can
    share the same group name to indicate they belong to the same logical group.
    """

    type: Literal["everest_parameters"] = "everest_parameters"
    dimensionality: Literal[1] = 1
    input_key: str
    forward_init: bool = False
    output_file: str = ""
    forward_init_file: str = ""
    update: bool = False
    control_type_: Literal["well_control", "generic_control"]
    initial_guess: float
    variable_type: Literal["real", "integer"]
    enabled: bool
    min: float
    max: float
    perturbation_type: Literal["absolute", "relative"]
    perturbation_magnitude: float
    scaled_range: tuple[float, float]
    sampler: SamplerConfig | None

    # Optional reference to the control group this variable belongs to
    group: str | None = None

    # Set up for deprecation, but has to live here until support for the
    # "dotdash" notation is removed for everest controls via everest config.
    input_key_dotdash: str = ""

    @property
    def parameter_keys(self) -> list[str]:
        return [self.input_key]

    @property
    def cardinality(self) -> ParameterCardinality:
        return ParameterCardinality.multiple_configs_per_ensemble_dataset

    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> xr.Dataset:
        raise NotImplementedError

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def load_parameter_graph(self) -> nx.Graph[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return 1

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> dict[str, dict[str, float | str]] | None:
        """Load this control's parameter value.

        Returns a dict suitable for aggregation with other controls in the same group.
        The actual file writing is handled at a higher level after all controls in
        a group have been collected.
        """
        df = ensemble.load_parameters(self.name, real_nr)
        assert isinstance(df, pl.DataFrame)

        # Extract the single value for this control
        value = df[self.input_key].item()

        # Return in a format that can be aggregated
        # The key structure supports nested keys like "point.x" -> {"x": value}
        # or "point.0.x" -> {"0": {"x": value}}
        key_without_group = (
            self.input_key.replace(f"{self.group}.", "", 1)
            if self.group
            else self.input_key
        )
        return {self.group or self.name: {key_without_group: value}}

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[None, pl.DataFrame]]:
        df = pl.DataFrame(
            {
                "realization": iens_active_index,
                self.input_key: pl.Series(from_data.flatten()),
            },
            strict=False,
        )
        yield None, df
