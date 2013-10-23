from ert.enkf.plot.data import DictProperty, SampleList


class PlotData(dict):
    observations = DictProperty("observations")
    refcase = DictProperty("refcase")
    ensemble = DictProperty("ensemble")
    name = DictProperty("name")

    min_x = DictProperty("min_x")
    max_x = DictProperty("max_x")
    min_value = DictProperty("min_value")
    max_value = DictProperty("max_value")

    def __init__(self):
        super(PlotData, self).__init__()

        self.observations = None
        self.refcase = None
        self.ensemble = None
        self.name = None

        self.min_x = None
        self.max_x = None
        self.min_value = None
        self.max_value = None


    def setRefcase(self, refcase):
        assert isinstance(refcase, SampleList)

        self.refcase = refcase

        if self.min_x is None:
            self.min_x = refcase.min_x
        else:
            self.min_x = min(self.min_x, refcase.min_x)

        if self.max_x is None:
            self.max_x = refcase.max_x
        else:
            self.max_x = max(self.max_x, refcase.max_x)

        if self.min_value is None:
            self.min_value = refcase.statistics.min_value
        else:
            self.min_value = min(self.min_value, refcase.statistics.min_value)

        if self.max_value is None:
            self.max_value = refcase.statistics.max_value
        else:
            self.max_value = max(self.max_value, refcase.statistics.max_value)

    def setObservations(self, observations):
        assert isinstance(observations, SampleList)

        self.observations = observations

        if self.min_x is None:
            self.min_x = observations.min_x
        else:
            self.min_x = min(self.min_x, observations.min_x)

        if self.max_x is None:
            self.max_x = observations.max_x
        else:
            self.max_x = max(self.max_x, observations.max_x)

        if self.min_value is None:
            self.min_value = self.adjustMinValue(observations.statistics.min_value, observations.statistics.min_with_std)
        else:
            mv = self.adjustMinValue(observations.statistics.min_value, observations.statistics.min_with_std)
            self.min_value = min(self.min_value, mv)

        if self.max_value is None:
            self.max_value = observations.statistics.max_with_std
        else:
            self.max_value = max(self.max_value, observations.statistics.max_with_std)


    def adjustMinValue(self, value, value_with_std):
        if value >= 0:
            return max(0, value_with_std)

        return value_with_std



