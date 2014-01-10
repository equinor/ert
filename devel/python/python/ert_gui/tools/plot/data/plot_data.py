from PyQt4.QtCore import QObject, pyqtSlot, QString
from ert_gui.tools.plot.data import HistogramPlotData, ObservationPlotData, RefcasePlotData, EnsemblePlotData


class PlotData(QObject):
    def __init__(self, name, parent=None):
        QObject.__init__(self, parent)

        self.__name = name

        #: :type: ObservationPlotData
        self.__observation_data = None
        #: :type: RefcasePlotData
        self.__refcase_data = None
        #: :type: EnsemblePlotData
        self.__ensemble_data = {}

        self.__histogram_data = {}


        self.__min_x = None
        self.__max_x = None
        self.__min_y = None
        self.__max_y = None

        self.__case_list = []


    def setObservationData(self, observation_data):
        observation_data.setParent(self)
        self.__observation_data = observation_data
        self.updateBoundaries(observation_data.minX(), observation_data.maxX(), observation_data.minY(), observation_data.maxY())


    def setRefcaseData(self, refcase_data):
        refcase_data.setParent(self)
        self.__refcase_data = refcase_data
        self.updateBoundaries(refcase_data.minX(), refcase_data.maxX(), refcase_data.minY(), refcase_data.maxY())


    def addEnsembleData(self, ensemble_data):
        ensemble_data.setParent(self)
        case_name = ensemble_data.caseName()
        self.__case_list.append(case_name)
        self.__ensemble_data[case_name] = ensemble_data
        self.updateBoundaries(ensemble_data.minX(), ensemble_data.maxX(), ensemble_data.minY(), ensemble_data.maxY())


    def updateBoundaries(self, min_x, max_x, min_y, max_y):
        if min_x is not None and (self.__min_x is None or self.__min_x > min_x):
            self.__min_x = min_x

        if max_x is not None and (self.__max_x is None or self.__max_x < max_x):
            self.__max_x = max_x

        if min_y is not None and (self.__min_y is None or self.__min_y > min_y):
            self.__min_y = min_y

        if max_y is not None and (self.__max_y is None or self.__max_y < max_y):
            self.__max_y = max_y



    @pyqtSlot(result=str)
    def name(self):
        """ @rtype: str """
        return self.__name

    @pyqtSlot(result=QObject)
    def observationData(self):
        """ @rtype: ObservationPlotData """
        return self.__observation_data

    @pyqtSlot(result=bool)
    def hasObservationData(self):
        """ @rtype: bool """
        return self.__observation_data is not None and self.__observation_data.isValid()

    @pyqtSlot(result=QObject)
    def refcaseData(self):
        """ @rtype: RefcasePlotData """
        return self.__refcase_data

    @pyqtSlot(result=bool)
    def hasRefcaseData(self):
        """ @rtype: bool """
        return self.__refcase_data is not None and self.__refcase_data.isValid()

    @pyqtSlot(QString, result=QObject)
    def ensembleData(self, case_name):
        """ @rtype: EnsemblePlotData """
        return self.__ensemble_data[str(case_name)]

    @pyqtSlot(result=bool)
    def hasEnsembleData(self):
        """ @rtype: bool """
        return len(self.__ensemble_data) > 0

    @pyqtSlot(QString, result=bool)
    def hasEnsembleDataForCase(self, case_name):
        """ @rtype: bool """
        return str(case_name) in self.__ensemble_data

    @pyqtSlot(QString, result=int)
    def realizationCount(self, case):
        """ @rtype: int """
        return self.__ensemble_data[str(case)].realizationCount()


    @pyqtSlot(result=float)
    def minX(self):
        return self.__min_x

    @pyqtSlot(result=float)
    def maxX(self):
        return self.__max_x

    @pyqtSlot(result=float)
    def minY(self):
        return self.__min_y

    @pyqtSlot(result=float)
    def maxY(self):
        return self.__max_y

    @pyqtSlot(result=int)
    def lastReportStepTime(self):
        last_report_step_time = 0

        if self.hasObservationData():
            last_report_step_time = self.__observation_data.lastReportStepTime()

        if self.hasRefcaseData():
            last_report_step_time = max(last_report_step_time, self.__refcase_data.lastReportStepTime())

        if self.hasEnsembleData():

            for case_name in self.__case_list:
                ensemble_data = self.ensembleData(case_name)
                last_report_step_time = max(last_report_step_time, ensemble_data.lastReportStepTime())

        return last_report_step_time


    @pyqtSlot(result=bool)
    def isValid(self):
        return self.hasBoundaries() and (self.hasObservationData() or self.hasRefcaseData() or self.hasEnsembleData())

    @pyqtSlot(result=bool)
    def hasBoundaries(self):
        return self.__min_x is not None and self.__max_x is not None and self.__min_y is not None and self.__max_y is not None

    @pyqtSlot(result="QStringList")
    def caseList(self):
        return self.__case_list


    @pyqtSlot(int, result=QObject)
    def histogramData(self, report_step_time):
        if not report_step_time in self.__histogram_data:
            histogram_data = HistogramPlotData(self.__name, report_step_time, parent=self)

            if self.hasObservationData():
                obs_data = self.observationData()
                if obs_data.hasSample(report_step_time):
                    histogram_data.setObservation(obs_data.getSample(report_step_time), obs_data.getError(report_step_time))

            if self.hasRefcaseData():
                refcase_data = self.refcaseData()
                if refcase_data.hasSample(report_step_time):
                    histogram_data.setRefcase(refcase_data.getSample(report_step_time))


            for case_name in self.__case_list:
                ensemble_data = self.ensembleData(case_name)

                if ensemble_data.hasSample(report_step_time) and not histogram_data.hasCaseHistogram(case_name):
                    samples = ensemble_data.getSample(report_step_time)
                    for sample in samples:
                        histogram_data.addSample(case_name, sample)

            self.__histogram_data[report_step_time] = histogram_data

        return self.__histogram_data[report_step_time]
