from ._base import BaseStorage
from concurrent.futures import Executor
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd


class PdHdf5(BaseStorage[npt.NDArray[np.float64]]):
    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        with pd.HDFStore(self.path / "params.h5", mode="a") as store:
            store.put(name, pd.DataFrame(array))

    def save_response(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        with pd.HDFStore(self.path / f"real_{iens}.h5", mode="a") as store:
            store.put(name, pd.DataFrame(array))

    def save_response_mt(
        self, name: str, array: npt.NDArray[np.float64], iens: int, executor: Executor
    ) -> None:
        executor.submit(self.save_response, name, array, iens)


class PdHdf5Open(BaseStorage[npt.NDArray[np.float64]]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self._stores: Dict[int, pd.HDFStore] = {}
        self._param_store = pd.HDFStore(self.path / "params.h5", mode="a")

    def __del__(self):
        for store in self._stores.values():
            store.close()
        self._param_store.close()

    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        self._param_store.put(name, pd.DataFrame(array))

    def save_parameter_mt(
        self, name: str, array: npt.NDArray[np.float64], executor: Executor
    ) -> None:
        executor.submit(self._param_store.put, name, pd.DataFrame(array))

    def save_response(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        if iens in self._stores:
            store = self._stores[iens]
        else:
            store = pd.HDFStore(self.path / f"real_{iens}.h5", mode="a")
            self._stores[iens] = store
        store.put(name, pd.DataFrame(array))
