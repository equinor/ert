from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import polars as pl

from ert.substitutions import substitute_runpath_name

from .gen_kw_config import DataSource, GenKwConfig
from .parameter_config import ParameterMetadata

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

    Number = int | float
    DataType = Mapping[str, Number | Mapping[str, Number]]
    MutableDataType = MutableMapping[str, Number | MutableMapping[str, Number]]


class ExtParamConfig(GenKwConfig):
    """Create an ExtParamConfig for @key with the given @input_keys

    @input_keys can be either a list of keys as strings or a dict with
    keys as strings and a list of suffixes for each key.
    If a list of strings is given, the order is preserved.
    
    This class extends GenKwConfig to use polars.DataFrame for scalar-like
    parameters instead of xarray.Dataset, providing better integration with
    ensemble-level data operations and storage.
    """

    type: Literal["everest_parameters"] = "everest_parameters"  # type: ignore[assignment]
    input_keys: list[str] = field(default_factory=list)
    distribution: None = None  # type: ignore[assignment]  # ExtParamConfig doesn't use distributions
    forward_init: bool = False
    output_file: str = ""
    update: bool = False
    input_source: DataSource = DataSource.DESIGN_MATRIX  # External parameters come from design/optimization

    @property
    def parameter_keys(self) -> list[str]:
        return self.input_keys

    @property
    def metadata(self) -> list[ParameterMetadata]:
        return []

    def __len__(self) -> int:
        return len(self.input_keys)

    def read_from_runpath(
        self, run_path: Path, real_nr: int, iteration: int
    ) -> pl.DataFrame:
        raise NotImplementedError

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> dict[str, dict[str, float | str]]:
        file_path = run_path / substitute_runpath_name(
            self.output_file, real_nr, ensemble.iteration
        )
        Path.mkdir(file_path.parent, exist_ok=True, parents=True)

        # Load parameters as polars DataFrame
        df = ensemble.load_parameters(self.name, real_nr)
        
        assert isinstance(df, pl.DataFrame)
        
        # Build hierarchical data structure from DataFrame
        data: MutableDataType = {}
        for col in df.columns:
            if col == "realization":
                continue
            value = df[col][0]
            
            # Handle nested structure with null byte separator
            if "\0" in col:
                outer, inner = col.split("\0")
                if outer not in data:
                    data[outer] = {}
                data[outer][inner] = float(value)  # type: ignore
            else:
                data[col] = float(value)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        return {self.name: data}  # type: ignore

    def create_storage_datasets(
        self,
        from_data: npt.NDArray[np.float64],  # type: ignore[name-defined]
        iens_active_index: npt.NDArray[np.int_],  # type: ignore[name-defined]
    ) -> Iterator[tuple[int | None, pl.DataFrame]]:
        # Create column names with null byte separator for nested structure
        column_names = [
            x.split(f"{self.name}.")[1].replace(".", "\0")
            if f"{self.name}." in x
            else x
            for x in self.parameter_keys
        ]
        
        # Yield single DataFrame with all realizations (ensemble-level)
        df_data = {"realization": iens_active_index}
        for i, col_name in enumerate(column_names):
            df_data[col_name] = pl.Series(from_data[i, :])
        
        yield (None, pl.DataFrame(df_data))

    def load_parameters(
        self, ensemble: Ensemble, realizations: npt.NDArray[np.int_]  # type: ignore[name-defined]
    ) -> npt.NDArray[np.float64]:  # type: ignore[name-defined]
        """Load parameters from ensemble as numpy array.
        
        Returns a 2D array where each column is a realization and each row
        is a parameter value.
        """
        df = ensemble.load_parameters(self.name, realizations)
        assert isinstance(df, pl.DataFrame)
        return df.drop("realization").to_numpy().T.copy()

    def load_parameter_graph(self) -> nx.Graph[int]:
        """Return independence graph (no edges) since parameters are independent."""
        graph_independence: nx.Graph[int] = nx.Graph()
        graph_independence.add_nodes_from(range(len(self.input_keys)))
        return graph_independence
