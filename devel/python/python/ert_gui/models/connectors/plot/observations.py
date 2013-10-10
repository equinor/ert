from sys import float_info
from ert.enkf.enums import EnkfObservationImplementationType
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ModelMixin



class DictProperty(object):
    def __init__(self, name):
        super(DictProperty, self).__init__()
        self.__property_name = name

    def __set__(self, instance, value):
        instance[self.__property_name] = value

    def __get__(self, instance, owner):
        if not self.__property_name in instance:
            raise AttributeError("The dictionary property: '%s' has not been initialized!" % self.__property_name)

        if not owner.__dict__.has_key(self.__property_name):
            raise AttributeError("The dictionary property: '%s' does not have an associated attribute!" % self.__property_name)

        return instance[self.__property_name]


class Sample(dict):
    index = DictProperty("index")
    x = DictProperty("x")
    value = DictProperty("value")
    std = DictProperty("std")

    group = DictProperty("group")
    name = DictProperty("name")
    single_point = DictProperty("single_point")

    def __init__(self):
        super(Sample, self).__init__()

        self.index = None
        self.x = 0.0
        self.value = 0.0
        self.std = 0.0

        self.group = None
        self.name = None

        self.single_point = False


class SampleStatistics(dict):
    min_x = DictProperty("min_x")
    max_x = DictProperty("max_x")

    min_value = DictProperty("min_value")
    max_value = DictProperty("max_value")

    min_std = DictProperty("min_std")
    max_std = DictProperty("max_std")

    def __init__(self):
        super(SampleStatistics, self).__init__()

        self.min_x = None
        self.max_x = None

        self.min_value = None
        self.max_value = None

        self.min_std = None
        self.max_std = 0.0


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

        if self.min_std is None:
            self.min_std = sample.std

        self.min_x = min(self.min_x, sample.x)
        self.max_x = max(self.max_x, sample.x)

        self.min_value = min(self.min_value, sample.value)
        self.max_value = max(self.max_value, sample.value)

        self.min_std = min(self.min_std, sample.std)
        self.max_std = max(self.max_std, sample.std)


class SampleList(dict):
    min_x = DictProperty("min_x")
    max_x = DictProperty("max_x")
    group = DictProperty("group")
    samples = DictProperty("samples")
    statistics = DictProperty("statistics")
    continuous_line = DictProperty("continuous_line")


    def __init__(self):
        super(SampleList, self).__init__()

        self.min_x = None
        self.max_x = None
        self.group = None
        self.samples = []
        self.statistics = SampleStatistics()
        self.continuous_line = True

    def addSample(self, sample):
        assert isinstance(sample, Sample)

        self.samples.append(sample)
        # self.samples.sort(key=Sample.index)
        self.statistics.addSample(sample)
        if sample.single_point:
            self.continuous_line = False

class SampleListCollection(dict):
    # sample_lists = DictProperty("sample_lists")
    sample_lists_keys = DictProperty("sample_lists_keys")

    def __init__(self):
        super(SampleListCollection, self).__init__()

        # self.sample_lists = []
        self.sample_lists_keys = []

    def addSampleList(self, sample_list):
        assert isinstance(sample_list, SampleList)

        if sample_list.group in self:
            raise ValueError("Already exists a list for group with name: %s" % sample_list.group)

        # self.sample_lists.append(sample_list)

        self.sample_lists_keys.append(sample_list.group)
        self.sample_lists_keys.sort()
        self[sample_list.group] = sample_list




class ObservationsModel(ErtConnector, ModelMixin):
    def __init__(self):
        super(ObservationsModel, self).__init__()

    def getObservationKeys(self):
        observations = self.ert().getObservations()
        keys = observations.getTypedKeylist(EnkfObservationImplementationType.SUMMARY_OBS)
        keys = sorted(keys)
        return keys

    def getObservations(self, key):
        """ @rtype: list of Sample """
        observations = self.ert().getObservations()
        assert observations.hasKey(key)
        observation_data = observations.getObservationsVector(key)
        active_count = observation_data.getActiveCount()

        result = []
        history_length = self.ert().getHistoryLength()
        for index in range(0, history_length):
            if observation_data.isActive(index):
                sample = Sample()
                sample.index = index
                sample.x = observations.getObservationTime(index).ctime()

                #: :type: SummaryObservation
                node = observation_data.getNode(index)

                sample.value = node.getValue()
                sample.std = node.getStandardDeviation()
                sample.group = node.getSummaryKey()
                sample.name = key

                if active_count == 1:
                    sample.single_point = True

                result.append(sample)

        return result

    def getAllObservations(self):
        keys = self.getObservationKeys()

        result = SampleListCollection()

        for key in keys:
            observations = self.getObservations(key)

            for observation in observations:
                if not observation.group in result:
                    sample_list = SampleList()
                    sample_list.group = observation.group
                    sample_list.min_x = self.getFirstReportStep()
                    sample_list.max_x = self.getLastReportStep()

                    result.addSampleList(sample_list)

                result[observation.group].addSample(observation)

        return result

    def getFirstReportStep(self):
        return self.ert().getObservations().getObservationTime(0).ctime()

    def getLastReportStep(self):
        history_length = self.ert().getHistoryLength()
        return self.ert().getObservations().getObservationTime(history_length - 1).ctime()


