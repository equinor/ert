from typing import Optional, Tuple, Callable

import numpy as np
import scipy.stats

import ert3


class Distribution:
    def __init__(
        self,
        *,
        size: Optional[int],
        index: Optional[Tuple[int, ...]],
        rvs: Callable[[int], np.ndarray],
        ppf: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        if size is None and index is None:
            raise ValueError("Cannot create distribution with neither size nor index")
        if size is not None and index is not None:
            raise ValueError("Cannot create distribution with both size and index")

        if size is not None:
            self._size = size
            self._as_array = True
            self._index = tuple(range(self._size))
        elif index is not None:
            self._size = len(tuple(index))
            self._as_array = False
            self._index = tuple(index)

        self._raw_rvs = rvs
        self._raw_ppf = ppf

    @property
    def index(self) -> Tuple[int, ...]:
        return self._index

    def _to_record(self, x: np.ndarray) -> ert3.data.Record:
        if self._as_array:
            return ert3.data.Record(data=x.tolist())
        else:
            return ert3.data.Record(
                data={idx: float(val) for idx, val in zip(self.index, x)}
            )

    def sample(self) -> ert3.data.Record:
        return self._to_record(self._raw_rvs(self._size))

    def ppf(self, x: float) -> ert3.data.Record:
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
        index: Optional[Tuple[int, ...]] = None
    ) -> None:
        self._mean = mean
        self._std = std

        def rvs(size: int) -> np.ndarray:
            return np.array(
                scipy.stats.norm.rvs(loc=self._mean, scale=self._std, size=size)
            )

        def ppf(x: np.ndarray) -> np.ndarray:
            return np.array(scipy.stats.norm.ppf(x, loc=self._mean, scale=self._std))

        super().__init__(
            size=size,
            index=index,
            rvs=rvs,
            ppf=ppf,
        )


class Uniform(Distribution):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        *,
        size: Optional[int] = None,
        index: Optional[Tuple[int, ...]] = None
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scale = upper_bound - lower_bound

        def rvs(size: int) -> np.ndarray:
            return np.array(
                scipy.stats.uniform.rvs(
                    loc=self._lower_bound, scale=self._scale, size=self._size
                )
            )

        def ppf(x: np.ndarray) -> np.ndarray:
            return np.array(
                scipy.stats.uniform.ppf(x, loc=self._lower_bound, scale=self._scale)
            )

        super().__init__(
            size=size,
            index=index,
            rvs=rvs,
            ppf=ppf,
        )
