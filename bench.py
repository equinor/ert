#!/usr/bin/env python3
import sys
from contextlib import contextmanager
from time import perf_counter
from res.enkf.enkf_fs import EnkfFs as _EnkfFs
from res._lib.enkf_fs import (
    write_param_vector_raw,
    read_param_vector_raw,
    write_resp_vector_raw,
)
from typing import Any, Dict, Generic, List, Tuple, Generator, Optional, TypeVar
from shutil import rmtree
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Executor, ThreadPoolExecutor
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd
import argparse


from abc import ABC, abstractmethod


RecordType = TypeVar("RecordType")


@contextmanager
def timer() -> Generator[None, None, None]:
    start = perf_counter()
    yield
    print(perf_counter() - start)


def skip():
    print("skip")
    sys.exit(0)


class BaseStorage(ABC, Generic[RecordType]):
    __use_threads__ = False

    @abstractmethod
    def save_parameter(self, name: str, array: RecordType) -> None:
        ...

    def save_parameter_mt(
        self, name: str, array: RecordType, executor: Executor
    ) -> None:
        skip()

    @abstractmethod
    def save_response(self, name: str, array: RecordType, iens: int) -> None:
        ...

    def save_response_mt(
        self, name: str, array: RecordType, iens: int, executor: Executor
    ) -> None:
        skip()

    @abstractmethod
    def from_numpy(self, array: npt.NDArray[np.float64]) -> RecordType:
        ...


class EnkfFs(BaseStorage[npt.NDArray[np.float64]]):
    __use_threads__ = True
    path: Path = Path("_tmp_enkf_fs")

    def __init__(self) -> None:
        rmtree(self.path, ignore_errors=True)
        self._fs = _EnkfFs.createFileSystem(str(self.path), mount=True)

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


class EnkfFsMt(BaseStorage[npt.NDArray[np.float64]]):
    path: Path = Path("_tmp_enkf_fs_mt")

    def __init__(self) -> None:
        rmtree(self.path, ignore_errors=True)
        self._fs = _EnkfFs.createFileSystem(str(self.path), mount=True)

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
        skip()


class PdHdf5(BaseStorage[npt.NDArray[np.float64]]):
    path: Path = Path("_tmp_pdhdf5")

    def __init__(self) -> None:
        rmtree(self.path, ignore_errors=True)
        self.path.mkdir()

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
    path: Path = Path("_tmp_pdhdf5open")

    def __init__(self) -> None:
        rmtree(self.path, ignore_errors=True)
        self.path.mkdir()

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


class XrCdf(BaseStorage[xr.DataArray]):
    path: Path = Path("_tmp_xarray_cdf")

    def __init__(self) -> None:
        rmtree(self.path, ignore_errors=True)
        self.path.mkdir()

        self._engine = "h5netcdf"

    def save_parameter(self, name: str, array: xr.DataArray) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / "params.nc", engine=self._engine, mode="a")

    def save_response(self, name: str, array: xr.DataArray, iens: int) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / f"real_{iens}.nc", mode="a", engine=self._engine)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> xr.DataArray:
        return xr.DataArray(array)


def gen_params(
    parameters: int, ensemble_size: int
) -> Generator[Tuple[str, npt.NDArray[np.float64]], None, None]:
    # rng = np.random.default_rng(seed=0)
    for i in range(parameters):
        yield f"TEST{i}", np.random.rand(ensemble_size, 10)


def gen_responses(
    responses: int, ensemble_size: int
) -> Generator[Tuple[int, str, npt.NDArray[np.float64]], None, None]:
    keys = [f"RESP{i}" for i in range(responses)]
    for iens in range(ensemble_size):
        for key in keys:
            yield iens, key, np.random.rand(10000)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    modules = []
    for key, val in globals().items():
        try:
            if val is not BaseStorage and issubclass(val, BaseStorage):
                modules.append(key)
        except:
            pass

    ap.add_argument("module", choices=modules)
    ap.add_argument("command", choices=BaseStorage.__abstractmethods__)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--parameters", type=int, default=10)
    ap.add_argument("--ensemble-size", type=int, default=100)

    return ap.parse_args()


def test_save_parameter(args: argparse.Namespace, storage: BaseStorage):
    with timer():
        for name, mat in gen_params(
            parameters=args.parameters, ensemble_size=args.ensemble_size
        ):
            storage.save_parameter(name, mat)


def test_save_parameter_mt(
    args: argparse.Namespace, storage: BaseStorage, executor: Executor
):
    with timer():
        for name, mat in gen_params(
            parameters=args.parameters, ensemble_size=args.ensemble_size
        ):
            storage.save_parameter_mt(name, mat, executor)

        executor.shutdown()


def test_save_response(args: argparse.Namespace, storage: BaseStorage):
    with timer():
        for iens, name, data in gen_responses(
            responses=10, ensemble_size=args.ensemble_size
        ):
            storage.save_response(name, data, iens)


def test_save_response_mt(
    args: argparse.Namespace, storage: BaseStorage, executor: Executor
):
    with timer():
        for iens, name, data in gen_responses(
            responses=10, ensemble_size=args.ensemble_size
        ):
            storage.save_response_mt(name, data, iens, executor)
        executor.shutdown()


def main() -> None:
    args = parse_args()

    storage = globals()[args.module]()

    kwargs: Dict[str, Any] = {}
    if args.threads > 1:
        command = f"test_{args.command}_mt"
        if storage.__use_threads__:
            kwargs["executor"] = ThreadPoolExecutor(max_workers=args.threads)
        else:
            kwargs["executor"] = ProcessPoolExecutor(max_workers=args.threads)
    else:
        command = f"test_{args.command}"

    globals()[command](args, storage, **kwargs)


if __name__ == "__main__":
    main()
