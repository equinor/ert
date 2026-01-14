from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import field
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, Self

import networkx as nx
import numpy as np
import polars as pl
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ropt.workflow import find_sampler_plugin

from ert.substitutions import substitute_runpath_name

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
    """Create an EverestControl for @key with the given @input_keys

    @input_keys can be either a list of keys as strings or a dict with
    keys as strings and a list of suffixes for each key.
    If a list of strings is given, the order is preserved.
    """

    type: Literal["everest_parameters"] = "everest_parameters"
    input_keys: list[str] = field(default_factory=list)
    forward_init: bool = False
    output_file: str = ""
    forward_init_file: str = ""
    update: bool = False
    types: list[Literal["well_control", "generic_control"]]
    initial_guesses: list[float]
    control_types: list[Literal["real", "integer"]]
    enabled: list[bool]
    min: list[float]
    max: list[float]
    perturbation_types: list[Literal["absolute", "relative"]]
    perturbation_magnitudes: list[float]
    scaled_ranges: list[tuple[float, float]]
    samplers: list[SamplerConfig | None]

    # Set up for deprecation, but has to live here until support for the
    # "dotdash" notation is removed for everest controls via everest config.
    input_keys_dotdash: list[str] = field(default_factory=list)

    @property
    def parameter_keys(self) -> list[str]:
        return self.input_keys

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
        return len(self.input_keys)

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> None:
        file_path: Path = run_path / substitute_runpath_name(
            self.output_file, real_nr, ensemble.iteration
        )
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)

        data: dict[str, Any] = {}
        df = ensemble.load_parameters(self.name, real_nr)
        assert isinstance(df, pl.DataFrame)
        df = df.drop("realization")
        df = df.rename({col: col.replace(f"{self.name}.", "", 1) for col in df.columns})
        for c in df.columns:
            if "." in c:
                top_key, sub_key = c.split(".", 1)
                if top_key not in data:
                    data[top_key] = {}
                data[top_key][sub_key] = df[c].item()
            else:
                data[c] = df[c].item()

        file_path.write_text(json.dumps(data), encoding="utf-8")

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],
        iens_active_index: npt.NDArray[np.int_],
    ) -> Iterator[tuple[int | None, pl.DataFrame]]:
        df = pl.DataFrame(
            from_data,
            strict=False,
            schema=self.parameter_keys,
            orient="row",
        )
        df = df.with_columns(pl.Series("realization", iens_active_index)).select(
            ["realization", *self.parameter_keys]
        )

        yield None, df
