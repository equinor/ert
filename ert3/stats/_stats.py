import random
from typing import Callable, List, Optional, cast

import numpy as np
import scipy.stats

import ert


class Distribution:
    """A distribution is an object that can draw stochastic numbers in the
    form of a NumericalRecord.

    The record is a collection with size, and either a dummy integer index or
    an explicit index. Either size or index must be provided. When an index
    is provided, the size is implicit from the length of the index.

    Every number in the sampled NumericalRecord (if size > 1) is independent.￼

    Args:
        size: Length of the NumericalRecord
        index: Explicit integer or string index if size is not provided.
        rvs: Callback to a function that draws the numbers.
        ppf: Callback to a function that computes the percentile function.
    """

    def __init__(
        self,
        *,
        size: Optional[int],
        index: Optional[ert.data.RecordIndex],
        rvs: Callable[[int], np.ndarray],  # type: ignore
        ppf: Callable[[np.ndarray], np.ndarray],  # type: ignore
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
            self._index = cast(ert.data.RecordIndex, tuple(idx for idx in index))

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
                data={
                    idx: float(val) for idx, val in zip(self.index, x)  # type: ignore
                }
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
        index: Optional[ert.data.RecordIndex] = None,
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
        index: Optional[ert.data.RecordIndex] = None,
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


class Discrete(Distribution):
    """Draw a NumericalRecord of specified size (or with specified index) from
    a discrete list of values. Each value has equal weight.

    Only float values are supported.
    """

    def __init__(
        self,
        values: List[float],
        *,
        size: Optional[int] = None,
        index: Optional[ert.data.RecordIndex] = None,
    ) -> None:
        self._values = values
        self._sortedvalues = sorted(self._values)

        def rvs(size: int) -> np.ndarray:  # type: ignore
            return np.array(random.choices(self._values, k=size))

        def ppf(x: np.ndarray) -> np.ndarray:  # type: ignore
            # pylint: disable=line-too-long
            # See https://openpress.usask.ca/introtoappliedstatsforpsych/chapter/6-1-discrete-data-percentiles-and-quartiles/ # noqa: E501
            # and in particular equation 6.2 (keeping in mind zero-indexing in Python)
            n = len(self._sortedvalues)
            idxs = np.ceil(x * n).astype(int)
            retval: np.ndarray = np.array(  # type: ignore
                [self._sortedvalues[i - 1] if 1 <= i <= n else np.nan for i in idxs]
            )
            return retval

        super().__init__(
            size=size,
            index=index,
            rvs=rvs,
            ppf=ppf,
        )

    @property
    def values(self) -> List[float]:
        return self._values

    @property
    def type(self) -> str:
        return "discrete"


class Constant(Discrete):
    """A constant distribution will make a NumericalRecord of specified size or with
    specified index, but with all elements equal to the specified value"""

    def __init__(
        self,
        value: float,
        *,
        size: Optional[int] = None,
        index: Optional[ert.data.RecordIndex] = None,
    ) -> None:
        self._value = value

        super().__init__(
            [value],
            size=size,
            index=index,
        )

    @property
    def value(self) -> float:
        return self._value

    @property
    def type(self) -> str:
        return "constant"
