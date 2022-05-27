#!/usr/bin/env python3
import os
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
import argparse
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd
import sqlalchemy.orm
import ert_storage.database_schema as ds
import ert_storage.database


from abc import ABC, abstractmethod


@contextmanager
def timer() -> Generator[None, None, None]:
    start = perf_counter()
    yield
    print(perf_counter() - start)


def skip():
    print("skip")
    sys.exit(0)


RecordType = TypeVar("RecordType")


class BaseStorage(ABC, Generic[RecordType]):
    __use_threads__ = False

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.path = Path.cwd() / f"_tmp_{self.__class__.__name__}"
        rmtree(self.path, ignore_errors=True)
        self.path.mkdir()
        os.chdir(self.path)

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

    def gen_params(self) -> List[Tuple[str, RecordType]]:
        # rng = np.random.default_rng(seed=0)
        return [
            (f"TEST{i}", self.from_numpy(np.random.rand(self.args.ensemble_size, 10)))
            for i in range(self.args.keys)
        ]

    def gen_responses(self) -> List[Tuple[int, str, RecordType]]:
        keys = [f"RESP{i}" for i in range(self.args.keys)]
        return [
            (iens, key, self.from_numpy(np.random.rand(10000)))
            for iens in range(self.args.ensemble_size)
            for key in keys
        ]

    def test_save_parameter(self) -> None:
        params = self.gen_params()
        with timer():
            for name, mat in params:
                self.save_parameter(name, mat)

    def test_save_parameter_mt(self, executor: Executor) -> None:
        params = self.gen_params()
        with timer():
            for name, mat in params:
                self.save_parameter_mt(name, mat, executor)
            executor.shutdown()

    def test_save_response(self) -> None:
        responses = self.gen_responses()
        with timer():
            for iens, name, data in responses:
                self.save_response(name, data, iens)

    def test_save_response_mt(self, executor: Executor) -> None:
        responses = self.gen_responses()
        with timer():
            for iens, name, data in responses:
                self.save_response_mt(name, data, iens, executor)
            executor.shutdown()


class EnkfFs(BaseStorage[npt.NDArray[np.float64]]):
    __use_threads__ = True

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
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
        skip()


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


class XrCdf(BaseStorage[xr.DataArray]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self._engine = "h5netcdf"

    def save_parameter(self, name: str, array: xr.DataArray) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / "params.nc", engine=self._engine, mode="a")

    def save_response(self, name: str, array: xr.DataArray, iens: int) -> None:
        da = xr.Dataset({name: array})
        da.to_netcdf(self.path / f"real_{iens}.nc", mode="a", engine=self._engine)

    def from_numpy(self, array: npt.NDArray[np.float64]) -> xr.DataArray:
        return xr.DataArray(array)


class Sqlite(BaseStorage[npt.NDArray[np.float64]]):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        ert_storage.database.Base.metadata.create_all(bind=ert_storage.database.engine)

        with self._session() as db:
            ensemble = ds.Ensemble(
                parameter_names=[],
                response_names=[],
                experiment=ds.Experiment(name="benchmark"),
                size=args.ensemble_size,
            )

            db.add(ensemble)
            db.refresh(ensemble)
            self._ensemble_id = ensemble.id

    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        with self._session() as db:
            record = ds.Record()

    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array

    def _new_record(
        self, db: sqlalchemy.orm.Session, name: str, realization_index: int
    ) -> ds.Record:
        ensemble = db.query(ds.Ensemble).filter_by(id=self._ensemble_id).one()

        q = (
            db.query(ds.Record)
            .join(ds.RecordInfo)
            .filter_by(ensemble_pk=ensemble.pk, name=name)
        )
        if (
            ensemble.size != -1
            and realization_index is not None
            and realization_index not in ensemble.active_realizations
        ):
            raise RuntimeError(
                f"Realization index {realization_index} outside "
                f"of allowed realization indices {ensemble.active_realizations}"
            )
        q = q.filter(
            (ds.Record.realization_index == None)
            or (ds.Record.realization_index == realization_index)
        )

        if q.count() > 0:
            raise RuntimeError(
                f"Ensemble-wide record '{name}' for ensemble '{self._ensemble_id}' already exists"
            )

        return ds.Record(
            record_info=ds.RecordInfo(ensemble=ensemble, name=name),
            realization_index=realization_index,
        )

    @contextmanager
    def _session(self) -> Generator[sqlalchemy.orm.Session, None, None]:
        with ert_storage.database.Session() as db:
            try:
                yield db
                db.commit()
                db.close()
            except:
                db.rollback()
                db.close()
                raise


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
    ap.add_argument("--keys", type=int, default=10)
    ap.add_argument("--ensemble-size", type=int, default=100)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    storage = globals()[args.module](args)

    kwargs: Dict[str, Any] = {}
    if args.threads > 1:
        command = f"test_{args.command}_mt"
        if storage.__use_threads__:
            kwargs["executor"] = ThreadPoolExecutor(max_workers=args.threads)
        else:
            kwargs["executor"] = ProcessPoolExecutor(max_workers=args.threads)
    else:
        command = f"test_{args.command}"

    getattr(storage, command)(**kwargs)


if __name__ == "__main__":
    main()
