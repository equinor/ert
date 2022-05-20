#!/usr/bin/env python3
import res
from res.enkf.enkf_fs import EnkfFs
from res._lib.enkf_fs import write_param_vector_raw, read_param_vector_raw
from typing import Dict, List, Tuple, Generator
from timeit import Timer
from p_tqdm import p_map
from shutil import rmtree
from pathlib import Path
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import numpy.typing as npt


print(f"{res.ResPrototype.lib=}")


from abc import ABC, abstractmethod


class BaseStorage(ABC):
    @abstractmethod
    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        ...


class EnkfFsStorage(BaseStorage):
    path: Path = Path("_tmp_enkf_fs")

    def __init__(self) -> None:
        rmtree(self.path)
        self._fs = EnkfFs.createFileSystem(str(self.path), mount=True)

    def save_parameter(self, name: str, array: npt.NDArray[np.float64]) -> None:
        with ThreadPoolExecutor(max_workers=2) as exec:
            def fn(x: Tuple[int, npt.NDArray[np.float64]]) -> None:
                iens = x[0]
                data = x[1]
                write_param_vector_raw(self._fs, data, name, iens)

            list(exec.map(fn, enumerate(array)))


def gen_params(parameters: int, ensemble_size: int) -> Generator[Tuple[str, npt.NDArray[np.float64]], None, None]:
    # rng = np.random.default_rng(seed=0)
    for i in range(parameters):
        yield f"TEST{i}", np.random.rand(ensemble_size, 10000)


def main() -> None:
    storage = EnkfFsStorage()
    for name, mat in gen_params(parameters=1, ensemble_size=100):
        storage.save_parameter(name, mat)


if __name__ == "__main__":
    print(f"{Timer(main).timeit(number=10)=}")
