import numpy as np
import numpy.typing as npt
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Tuple, Optional, List

from res.enkf.enkf_fs import EnkfFs as _EnkfFs
from res._lib.enkf_fs import (
    write_param_vector_raw,
    write_resp_vector_raw,
    load_resp_vector_raw,
)
from ._base import BaseStorage, Namespace


class EnkfFs(BaseStorage[npt.NDArray[np.float64]]):
    __use_threads__ = True

    def __init__(self, args: Namespace, keep: bool) -> None:
        super().__init__(args, keep)
        self._fs = _EnkfFs.createFileSystem(str(self.path), mount=True)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array

    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        for iens, data in enumerate(array):
            write_param_vector_raw(self._fs, data, name, iens)

    def save_parameter_mt(
        self, name: str, array: npt.NDArray[np.float64], executor: Executor
    ) -> None:
        def fn(x: Tuple[int, npt.NDArray[np.float64]]) -> None:
            iens = x[0]
            data = x[1]
            write_param_vector_raw(self._fs, data, name, iens)

        executor.map(fn, enumerate(array))

    def save_response(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        write_resp_vector_raw(self._fs, array, name, iens)

    def save_response_mt(
        self, name: str, array: npt.NDArray[np.float64], iens: int, executor: Executor
    ) -> None:
        executor.submit(write_resp_vector_raw, self._fs, array, name, iens)

    def load_response(
        self, name: str, iens: Optional[List[int]]
    ) -> npt.NDArray[np.float64]:
        if iens is None:
            return np.array(
                [
                    load_resp_vector_raw(self._fs, name, i)
                    for i in range(self.args.ensemble_size)
                ]
            )
        return np.array([load_resp_vector_raw(self._fs, name, i) for i in iens])


class EnkfFsMt(EnkfFs):
    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        def fn(x: Tuple[int, npt.NDArray[np.float64]]) -> None:
            iens = x[0]
            data = x[1]
            write_param_vector_raw(self._fs, data, name, iens)

        with ThreadPoolExecutor() as exec:
            list(exec.map(fn, enumerate(array)))

    def save_response(
        self, name: str, array: npt.NDArray[np.float64], iens: int
    ) -> None:
        self.skip()
