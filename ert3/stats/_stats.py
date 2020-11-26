import scipy.stats


class Gaussian:
    def __init__(self, mean, std, *, size=None, index=None):
        self._mean = mean
        self._std = std

        if size is None and index is None:
            raise ValueError(
                "Cannot create gaussian distribution with neither size nor index"
            )
        if size is not None and index is not None:
            raise ValueError(
                "Cannot create gaussian distribution with both size and index"
            )

        if size is not None:
            self._size = size
        else:
            self._size = len(tuple(index))
        self._index = index

    def sample(self):
        sample = scipy.stats.norm.rvs(loc=self._mean, scale=self._std, size=self._size)
        if self._index is None:
            return sample
        else:
            return {idx: float(val) for idx, val in zip(self._index, sample)}


class Uniform:
    def __init__(self, lower_bound, upper_bound, *, size=None, index=None):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scale = upper_bound - lower_bound

        if size is None and index is None:
            raise ValueError(
                "Cannot create uniform distribution with neither size nor index"
            )
        if size is not None and index is not None:
            raise ValueError(
                "Cannot create uniform distribution with both size and index"
            )

        if size is not None:
            self._size = size
        else:
            self._size = len(tuple(index))
        self._index = index

    def sample(self):
        sample = scipy.stats.uniform.rvs(
            loc=self._lower_bound, scale=self._scale, size=self._size
        )
        if self._index is None:
            return sample
        else:
            return {idx: float(val) for idx, val in zip(self._index, sample)}
