from typing import Optional, Callable

import numpy as np
import scipy.stats

import ert


class Distribution:
    def __init__(
        self,
        *,
        size: Optional[int],
        index: Optional[ert.data.RecordIndex],
        rvs: Callable[[int], np.ndarray],  # type: ignore
        ppf: Callable[[np.ndarray], np.ndarray]  # type: ignore
    ) -> None:
        if size is None and index is None:
            raise ValueError("Cannot create distribution with neither size nor index")
        if size is not None and index is not None:
            raise ValueError("Cannot create distribution with both size and index")

        self._index: ert.data.RecordIndex = ()

        if size is not None:
            self._size = size
            self._as_array = True
            self._index = tuple(range(self._size))
        elif index is not None:
            self._size = len(tuple(index))
            self._as_array = False
            self._index = tuple(idx for idx in index)

        self._raw_rvs = rvs
        self._raw_ppf = ppf

    @property
    def size(self) -> int:
        return self._size

    @property
    def index(self) -> ert.data.RecordIndex:
        return self._index

    def _to_record(self, x: np.ndarray) -> ert.data.Record:  # type: ignore
        if self._as_array:
            return ert.data.NumericalRecord(data=x.tolist())
        else:
            return ert.data.NumericalRecord(
                data={idx: float(val) for idx, val in zip(self.index, x)}
            )

    def sample(self) -> ert.data.Record:
        return self._to_record(self._raw_rvs(self._size))

    def ppf(self, x: float) -> ert.data.Record:
        x_array = np.full(self._size, x)
        result = self._raw_ppf(x_array)
        return self._to_record(result)


class Gaussian(Distribution):
    def __init__(
        self,
        mean: float,
        std: float,
        *,
        size: Optional[int] = None,
        index: Optional[ert.data.RecordIndex] = None
    ) -> None:
        self._mean = mean
        self._std = std

        def rvs(size: int) -> np.ndarray:  # type: ignore
            return np.array(
                scipy.stats.norm.rvs(loc=self._mean, scale=self._std, size=size)
            )

        def ppf(x: np.ndarray) -> np.ndarray:  # type: ignore
            return np.array(scipy.stats.norm.ppf(x, loc=self._mean, scale=self._std))

        super().__init__(
            size=size,
            index=index,
            rvs=rvs,
            ppf=ppf,
        )

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @property
    def type(self) -> str:
        return "gaussian"


class Uniform(Distribution):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        *,
        size: Optional[int] = None,
        index: Optional[ert.data.RecordIndex] = None
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scale = upper_bound - lower_bound

        def rvs(size: int) -> np.ndarray:  # type: ignore
            return np.array(
                scipy.stats.uniform.rvs(
                    loc=self._lower_bound, scale=self._scale, size=self._size
                )
            )

        def ppf(x: np.ndarray) -> np.ndarray:  # type: ignore
            return np.array(
                scipy.stats.uniform.ppf(x, loc=self._lower_bound, scale=self._scale)
            )

        super().__init__(
            size=size,
            index=index,
            rvs=rvs,
            ppf=ppf,
        )

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def type(self) -> str:
        return "uniform"
