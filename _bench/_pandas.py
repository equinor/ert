from tables import group
from ._base import BaseStorage, Namespace
from concurrent.futures import Executor
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, Optional, List


class PdHdf5(BaseStorage[pd.DataFrame]):
    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        with pd.HDFStore(self.path / "params.h5", mode="a") as store:
            store.put(name, pd.DataFrame(array))

    def save_response(self, name: str, df: pd.DataFrame, iens: int) -> None:
        with pd.HDFStore(self.path / f"real_{iens}.h5", mode="a") as store:
            store.put(name, df)

    def save_response_mt(
        self, name: str, df: pd.DataFrame, iens: int, executor: Executor
    ) -> None:
        executor.submit(self.save_response, name, df, iens)

    def load_response(self, name: str, iens: Optional[List[int]]) -> pd.DataFrame:
        if iens is None:
            return pd.concat(
                (
                    pd.read_hdf(self.path / f"real_{i}.h5", key=name)
                    for i in range(self.args.ensemble_size)
                )
            )

    def from_numpy(self, array: npt.NDArray[np.float64]) -> pd.DataFrame:
        return pd.DataFrame(array)


class PdHdf5Open(BaseStorage[pd.DataFrame]):
    def __init__(self, args: Namespace, keep: bool) -> None:
        super().__init__(args, keep)

        self._stores: Dict[int, pd.HDFStore] = {}
        self._param_store = pd.HDFStore(self.path / "params.h5", mode="a")

    def __del__(self):
        for store in self._stores.values():
            store.close()
        self._param_store.close()

    def save_parameter(self, name: str, df: pd.DataFrame) -> None:
        self._param_store.put(name, df)

    def save_parameter_mt(
        self, name: str, array: npt.NDArray[np.float64], executor: Executor
    ) -> None:
        executor.submit(self._param_store.put, name, pd.DataFrame(array))

    def save_response(self, name: str, df: pd.DataFrame, iens: int) -> None:
        if iens in self._stores:
            store = self._stores[iens]
        else:
            store = pd.HDFStore(self.path / f"real_{iens}.h5", mode="a")
            self._stores[iens] = store
        store.put(name, df)

    def load_response(self, name: str, iens: Optional[List[int]]) -> pd.DataFrame:
        if iens is None:
            return pd.concat(
                self._load_response(name, i) for i in range(self.args.ensemble_size)
            )

    def _load_response(self, name: str, iens: int) -> pd.DataFrame:
        if iens in self._stores:
            store = self._stores[iens]
        else:
            store = pd.HDFStore(self.path / f"real_{iens}.h5", mode="a")
            self._stores[iens] = store
        return store.get(name)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> pd.DataFrame:
        return pd.DataFrame(array)
