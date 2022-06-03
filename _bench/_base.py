from abc import ABC, abstractmethod
from concurrent.futures import Executor
from pathlib import Path
from time import perf_counter
from typing import Generator, Generic, Sequence, Optional, Tuple, TypeVar
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
    suffix: str

    ensemble_size: int
    key_size: int
    keys: int
    threads: int
    use_async: bool
    trials: int


class BaseStorage(ABC, Generic[RecordType]):
    __use_threads__ = False

    def __init__(self, args: Namespace, keep: bool = False) -> None:
        self.args = args
        self.path = Path.cwd() / f"_tmp_{self.__class__.__name__}-{args.suffix}"
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

    async def save_response_async(
        self, name: str, array: RecordType, iens: int
    ) -> None:
        self.skip()

    def save_response_mt(
        self, name: str, array: RecordType, iens: int, executor: Executor
    ) -> None:
        self.skip()

    @abstractmethod
    def load_parameter(self, name: str) -> RecordType:
        ...

    @abstractmethod
    def load_response(self, name: str, iens: Optional[Sequence[int]]) -> RecordType:
        ...

    async def load_response_async(
        self, name: str, iens: Optional[Sequence[int]]
    ) -> RecordType:
        ...

    @abstractmethod
    def from_numpy(self, array: npt.NDArray[np.float64]) -> RecordType:
        ...

    @abstractmethod
    def to_numpy(self, array: RecordType) -> npt.NDArray[np.float64]:
        ...

    def gen_params(self) -> Sequence[Tuple[str, RecordType]]:
        # rng = np.random.default_rng(seed=0)
        return [
            (
                f"TEST{i}",
                self.from_numpy(
                    np.random.rand(self.args.ensemble_size, self.args.key_size)
                ),
            )
            for i in range(self.args.keys)
        ]

    def gen_responses(self) -> Sequence[Tuple[int, str, RecordType]]:
        keys = [f"RESP{i}" for i in range(self.args.keys)]
        return [
            (iens, key, self.from_numpy(np.random.rand(self.args.key_size)))
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
            *(
                self.save_response_async(name, mat, iens)
                for iens, name, mat in responses
            )
        )

    def test_load_response_async(self) -> None:
        for _ in self.timer():
            asyncio.run(self._test_load_response_async())

    async def _test_load_response_async(self) -> None:
        await asyncio.gather(
            *(self.load_response_async(name, None) for name in self._response_names)
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

    def test_validate_parameter(self):
        params = self.gen_params()
        for _ in self.timer():
            for name, mat in params:
                self.save_parameter(name, mat)
            for name, mat in params:
                lhs = self.to_numpy(mat)
                rhs = self.to_numpy(self.load_parameter(name))
                try:
                    assert (lhs == rhs).all()
                except:
                    print("--- EXPECTED ---")
                    print(repr(lhs))
                    print("---- ACTUAL ----")
                    print(repr(rhs))
                    raise

    def test_validate_response(self):
        responses = self.gen_responses()
        for _ in self.timer():
            for iens, name, mat in responses:
                self.save_response(name, mat, iens)
            for iens, name, mat in responses:
                lhs = self.to_numpy(mat)
                rhs = self.to_numpy(self.load_response(name, [iens]))
                try:
                    assert (lhs == rhs).all()
                except:
                    print("--- EXPECTED ---")
                    print(repr(lhs))
                    print("---- ACTUAL ----")
                    print(repr(rhs))
                    raise

    def test_validate_response_async(self) -> None:
        for _ in self.timer():
            asyncio.run(self._test_validate_response_async())

    async def _test_validate_response_async(self) -> None:
        responses = self.gen_responses()
        await asyncio.gather(
            *(
                self.save_response_async(name, mat, iens)
                for iens, name, mat in responses
            )
        )

        await asyncio.gather(
            *(
                self._test_validate_response_assert_async(iens, name, mat)
                for iens, name, mat in responses
            )
        )

    async def _test_validate_response_assert_async(self, iens, name, mat):
        lhs = self.to_numpy(mat)
        rhs = self.to_numpy(await self.load_response_async(name, [iens]))
        try:
            assert (lhs == rhs).all()
        except:
            print("--- EXPECTED ---")
            print(repr(lhs))
            print("---- ACTUAL ----")
            print(repr(rhs))
            raise



class NumpyBaseStorage(BaseStorage[npt.NDArray[np.float64]]):
    def to_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array

    def from_numpy(self, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array
