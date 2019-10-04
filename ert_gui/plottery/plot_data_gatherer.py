from pandas import DataFrame
from res.enkf.export import GenKwCollector, SummaryCollector, GenDataCollector, SummaryObservationCollector, \
    GenDataObservationCollector, CustomKWCollector
from ert_shared import ERT

class PlotDataGatherer(object):

    def __init__(self, dataGatherFunc, conditionFunc, refcaseGatherFunc=None, observationGatherFunc=None, historyGatherFunc=None):
        super(PlotDataGatherer, self).__init__()

        self._dataGatherFunction = dataGatherFunc
        self._conditionFunction = conditionFunc
        self._refcaseGatherFunction = refcaseGatherFunc
        self._observationGatherFunction = observationGatherFunc
        self._historyGatherFunc = historyGatherFunc

    def hasHistoryGatherFunction(self):
        """ :rtype: bool """
        return self._historyGatherFunc is not None

    def hasRefcaseGatherFunction(self):
        """ :rtype: bool """
        return self._refcaseGatherFunction is not None

    def hasObservationGatherFunction(self):
        """ :rtype: bool """
        return self._observationGatherFunction is not None

    def canGatherDataForKey(self, key):
        """ :rtype: bool """
        return self._conditionFunction(key)

    def gatherData(self, case, key):
        """ :rtype: pandas.DataFrame """
        if not self.canGatherDataForKey(key):
            raise UserWarning("Unable to gather data for key: %s" % key)

        return self._dataGatherFunction(case, key)

    def gatherRefcaseData(self, ert, key):
        """ :rtype: pandas.DataFrame """
        if not self.canGatherDataForKey(key) or not self.hasRefcaseGatherFunction():
            raise UserWarning("Unable to gather refcase data for key: %s" % key)

        return self._refcaseGatherFunction(key)

    def gatherObservationData(self, case, key):
        """ :rtype: pandas.DataFrame """
        if not self.canGatherDataForKey(key) or not self.hasObservationGatherFunction():
            raise UserWarning("Unable to gather observation data for key: %s" % key)

        return self._observationGatherFunction(case, key)

    def gatherHistoryData(self, case, key):
        """ :rtype: pandas.DataFrame """
        if not self.canGatherDataForKey(key) or not self.hasHistoryGatherFunction():
            raise UserWarning("Unable to gather history data for key: %s" % key)

        return self._historyGatherFunc(case, key)


    @staticmethod
    def gatherGenKwData(case, key):
        """ :rtype: pandas.DataFrame """
        data = GenKwCollector.loadAllGenKwData(case, [key])
        return data[key].dropna()

    @staticmethod
    def gatherSummaryData(case, key):
        """ :rtype: pandas.DataFrame """
        data = SummaryCollector.loadAllSummaryData(case, [key])
        if not data.empty:
            data = data.reset_index()

            if any(data.duplicated()):
                print("** Warning: The simulation data contains duplicate "
                      "timestamps. A possible explanation is that your "
                      "simulation timestep is less than a second.")
                data = data.drop_duplicates()


            data = data.pivot(index="Date", columns="Realization", values=key)

        return data #.dropna()

    @staticmethod
    def gatherSummaryRefcaseData(key):
        refcase = ERT.enkf_facade.get_refcase()

        if refcase is None or key not in refcase:
            return DataFrame()

        values = refcase.numpy_vector(key, report_only=False)
        dates = refcase.numpy_dates

        data = DataFrame(zip(dates, values), columns=['Date', key])
        data.set_index("Date", inplace=True)

        return data.iloc[1:]

    @staticmethod
    def gatherSummaryHistoryData(case, key):
        # create history key
        if ":" in key:
            head, tail = key.split(":", 2)
            key = "%sH:%s" % (head, tail)
        else:
            key = "%sH" % key

        data = PlotDataGatherer.gatherSummaryRefcaseData(key)
        if data.empty and case is not None:
            data = PlotDataGatherer.gatherSummaryData(case, key)

        return data

    @staticmethod
    def gatherSummaryObservationData(case, key):
        if ERT.enkf_facade.is_key_with_observations(key):
            return SummaryObservationCollector.loadObservationData(case, [key]).dropna()
        else:
            return DataFrame()


    @staticmethod
    def gatherGenDataData(case, key):
        """ :rtype: pandas.DataFrame """
        key, report_step = key.split("@", 1)
        report_step = int(report_step)
        try:
            data = GenDataCollector.loadGenData(case, key, report_step)
        except ValueError:
            data = DataFrame()

        return data.dropna() # removes all rows that has a NaN


    @staticmethod
    def gatherGenDataObservationData(case, key_with_report_step):
        """ :rtype: pandas.DataFrame """
        key, report_step = key_with_report_step.split("@", 1)
        report_step = int(report_step)

        obs_key = GenDataObservationCollector.getObservationKeyForDataKey(key, report_step)

        if obs_key is not None:
            obs_data = GenDataObservationCollector.loadGenDataObservations(case, obs_key)
            columns = {obs_key: key_with_report_step, "STD_%s" % obs_key: "STD_%s" % key_with_report_step}
            obs_data = obs_data.rename(columns=columns)
        else:
            obs_data = DataFrame()

        return obs_data.dropna()

    @staticmethod
    def gatherCustomKwData(case, key):
        """ :rtype: pandas.DataFrame """
        data = CustomKWCollector.loadAllCustomKWData(case, [key])[key]

        return data
