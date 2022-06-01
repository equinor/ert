from abc import ABC, abstractmethod
from concurrent.futures import Executor
from pathlib import Path
from time import perf_counter
from typing import Generator, Generic, List, Optional, Tuple, TypeVar
import numpy as np
import numpy.typing as npt
import os
import shutil
import sys
import asyncio


RecordType = TypeVar("RecordType")


class Namespace:
    command: str
    module: str

    ensemble_size: int
    keys: int
    threads: int
    use_async: bool
    trials: int


class BaseStorage(ABC, Generic[RecordType]):
    __use_threads__ = False

    def __init__(self, args: Namespace, keep: bool = False) -> None:
        self.args = args
        self.path = Path.cwd() / f"_tmp_{self.__class__.__name__}"
        if not keep:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path.mkdir()
        os.chdir(self.path)

    def timer(self) -> Generator[None, None, None]:
        for _ in range(self.args.trials):
            start = perf_counter()
            yield
            print(perf_counter() - start)

    @staticmethod
    def skip() -> None:
        print("skip")
        sys.exit(0)

    @abstractmethod
    def save_parameter(self, name: str, array: RecordType) -> None:
        ...

    async def save_parameter_async(self, name: str, array: RecordType) -> None:
        self.skip()

    def save_parameter_mt(
        self, name: str, array: RecordType, executor: Executor
    ) -> None:
        self.skip()

    @abstractmethod
    def save_response(self, name: str, array: RecordType, iens: int) -> None:
        ...

    async def save_response_async(self, name: str, array: RecordType, iens: int) -> None:
        self.skip()

    def save_response_mt(
        self, name: str, array: RecordType, iens: int, executor: Executor
    ) -> None:
        self.skip()

    @abstractmethod
    def load_response(self, name: str, iens: Optional[List[int]]) -> RecordType:
        ...

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
        for _ in self.timer():
            for name, mat in params:
                self.save_parameter(name, mat)

    def test_save_parameter_async(self) -> None:
        params = self.gen_params()
        for _ in self.timer():
            asyncio.run(self._test_save_parameter_async(params))

    async def _test_save_parameter_async(self, params) -> None:
        await asyncio.gather(
            *(self.save_parameter_async(name, mat) for name, mat in params)
        )

    def test_save_parameter_mt(self, executor: Executor) -> None:
        params = self.gen_params()
        for _ in self.timer():
            for name, mat in params:
                self.save_parameter_mt(name, mat, executor)
            executor.shutdown()

    def test_save_response(self) -> None:
        responses = self.gen_responses()
        for _ in self.timer():
            for iens, name, data in responses:
                self.save_response(name, data, iens)

    def test_save_response_async(self) -> None:
        responses = self.gen_responses()
        for _ in self.timer():
            asyncio.run(self._test_save_response_async(responses))

    async def _test_save_response_async(self, responses) -> None:
        await asyncio.gather(
            *(self.save_response_async(name, mat, iens) for iens, name, mat in responses)
        )

    def test_save_response_mt(self, executor: Executor) -> None:
        responses = self.gen_responses()
        for _ in self.timer():
            for iens, name, data in responses:
                self.save_response_mt(name, data, iens, executor)
            executor.shutdown()

    def test_load_response(self) -> None:
        for _ in self.timer():
            for name in self._response_names:
                self.load_response(name, None)

    @property
    def _response_names(self) -> Generator[str, None, None]:
        for i in range(self.args.keys):
            yield f"RESP{i}"
