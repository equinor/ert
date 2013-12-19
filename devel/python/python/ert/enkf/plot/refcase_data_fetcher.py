from ert.ecl import EclSum, EclSumVector, EclSumNode
from ert.enkf.enums import ErtImplType
from ert.enkf.plot.data_fetcher import DataFetcher


class RefcaseDataFetcher(DataFetcher):
    def __init__(self, ert):
        super(RefcaseDataFetcher, self).__init__(ert)
        self.report_times = {}


    def hasRefcase(self):
        """ @rtype: bool """
        return self.ert().eclConfig().hasRefcase()

    def getRefCase(self):
        """ @rtype: EclSum """
        return self.ert().eclConfig().getRefcase()

    def getSummaryKeys(self):
        """ @rtype: StringList """
        return self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)


    def fetchData(self, key, case=None):
        data = {"x": None,
                "y": None,
                "min_y": None,
                "max_y": None,
                "min_x": None,
                "max_x": None}

        if not self.hasRefcase():
            return data


        refcase = self.getRefCase()
        vector = refcase.get_vector(key, report_only=True)

        data["x"] = [None] * len(vector)
        data["y"] = [None] * len(vector)

        index = 0
        for node in vector:
            assert isinstance(node, EclSumNode)

            x_value = self.getReportStepTimeFromRefcase(refcase, node.report_step)
            data["x"][index] = int(x_value)

            if data["min_x"] is None or data["min_x"] > x_value:
                data["min_x"] = x_value

            if data["max_x"] is None or data["max_x"] < x_value:
                data["max_x"] = x_value


            value = node.value
            data["y"][index] = float(value)

            if data["min_y"] is None or data["min_y"] > value:
                data["min_y"] = value

            if data["max_y"] is None or data["max_y"] < value:
                data["max_y"] = value

            index += 1

        return data


    def getReportStepTimeFromRefcase(self, refcase, report_step):
        if not report_step in self.report_times:
            self.report_times[report_step] = EclSum.cNamespace().get_report_time(refcase, report_step).ctime()

        return self.report_times[report_step]









