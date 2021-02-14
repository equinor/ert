import numpy as np
import scipy.stats
from typing import Optional, Iterable, Hashable, Any

# TODO: This should be an ert3.data.Record
Record = Any

class Distribution:
    def __init__(
        self, *, size: Optional[int], index: Optional[Iterable[Hashable]]
    ) -> None:
        if size is None and index is None:
            raise ValueError("Cannot create distribution with neither size nor index")
        if size is not None and index is not None:
            raise ValueError("Cannot create distribution with both size and index")

        if index is not None:
            self._size = len(tuple(index))
            self._as_array = False
            self._index = tuple(index)
        elif size is not None:
            self._size = size
            self._as_array = True
            self._index = tuple(range(self._size))
        else:
            # Note: This should never happen due to the initial checking above,
            # however mypy does not get that...
            raise AssertionError("Cannot create distribution with both size and index")

    @property
    def index(self) -> Iterable[Hashable]:
        return self._index

    def _add_index(self, x):
        if self._as_array:
            return x
        else:
            return {idx: float(val) for idx, val in zip(self.index, x)}

    def sample(self) -> Record:
        raise NotImplementedError("Distributions must implement 'sample'")

    def ppf(self, x) -> Record:
        raise NotImplementedError("Distributions must implement 'sample'")


class Gaussian(Distribution):
    def __init__(
        self,
        mean: float,
        std: float,
        *,
        size: Optional[int] = None,
        index: Optional[Iterable[Hashable]] = None
    ) -> None:
        super().__init__(size=size, index=index)
        self._mean = mean
        self._std = std

    def sample(self) -> Record:
        sample = scipy.stats.norm.rvs(loc=self._mean, scale=self._std, size=self._size)
        return self._add_index(sample)

    def ppf(self, x: float) -> Record:
        x = np.full(self._size, x)
        result = scipy.stats.norm.ppf(x, loc=self._mean, scale=self._std)
        return self._add_index(result)


class Uniform(Distribution):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        *,
        size: Optional[int] = None,
        index: Optional[Iterable[Hashable]] = None
    ) -> None:
        super().__init__(size=size, index=index)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scale = upper_bound - lower_bound

    def sample(self) -> Record:
        sample = scipy.stats.uniform.rvs(
            loc=self._lower_bound, scale=self._scale, size=self._size
        )
        return self._add_index(sample)

    def ppf(self, x: float) -> Record:
        x = np.full(self._size, x)
        result = scipy.stats.uniform.ppf(x, loc=self._lower_bound, scale=self._scale)
        return self._add_index(result)
