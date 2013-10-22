from ert.enkf.plot.data import DictProperty, Sample


class SampleStatistics(dict):
    min_x = DictProperty("min_x")
    max_x = DictProperty("max_x")

    min_value = DictProperty("min_value")
    max_value = DictProperty("max_value")

    min_with_std = DictProperty("min_with_std")
    max_with_std = DictProperty("max_with_std")

    def __init__(self):
        super(SampleStatistics, self).__init__()

        self.min_x = None
        self.max_x = None

        self.min_value = None
        self.max_value = None

        self.min_with_std = None
        self.max_with_std = None


    def addSample(self, sample):
        assert isinstance(sample, Sample)

        if self.min_x is None:
            self.min_x = sample.x

        if self.max_x is None:
            self.max_x = sample.x

        if self.min_value is None:
            self.min_value = sample.value

        if self.max_value is None:
            self.max_value = sample.value

        if self.min_with_std is None:
            self.min_with_std = sample.value - sample.std

        if self.max_with_std is None:
            self.max_with_std = sample.value + sample.std

        self.min_x = min(self.min_x, sample.x)
        self.max_x = max(self.max_x, sample.x)

        self.min_value = min(self.min_value, sample.value)
        self.max_value = max(self.max_value, sample.value)

        self.min_with_std = min(self.min_with_std, sample.value - sample.std)
        self.max_with_std = max(self.max_with_std, sample.value + sample.std)