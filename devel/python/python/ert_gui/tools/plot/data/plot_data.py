from PyQt4.QtCore import QObject, pyqtSlot, QString


class PlotData(QObject):
    def __init__(self, name, parent=None):
        QObject.__init__(self, parent)

        self.__name = name
        self.__obs_x_values = []
        self.__obs_y_values = []
        self.__obs_std_values = []
        self.__obs_is_continuous = True
        self.__has_observation_data = False

        self.__refcase_x_values = []
        self.__refcase_y_values = []
        self.__has_refcase_data = False

        self.__ensemble_x_values = {}
        self.__ensemble_y_values = {}
        self.__ensemble_y_min_values = {}
        self.__ensemble_y_max_values = {}


        self.__min_x = None
        self.__max_x = None
        self.__min_y = None
        self.__max_y = None

        self.__case_list = []


    def setObservationData(self, x_values, y_values, std_values, continuous):
        if x_values is not None and y_values is not None and std_values is not None:
            self.__obs_x_values = x_values
            self.__obs_y_values = y_values
            self.__obs_std_values = std_values
            self.__obs_is_continuous = continuous
            self.__has_observation_data = True


    def setRefcaseData(self, x_values, y_values):
        if x_values is not None and y_values is not None:
            self.__refcase_x_values = x_values
            self.__refcase_y_values = y_values
            self.__has_refcase_data = True


    def setEnsembleData(self, case, x_values, y_values, y_min_values, y_max_values):
        if x_values is not None and y_values is not None and y_min_values is not None and y_max_values is not None:
            self.__ensemble_x_values[case] = x_values
            self.__ensemble_y_values[case] = y_values

            self.__ensemble_y_min_values[case] = y_min_values
            self.__ensemble_y_max_values[case] = y_max_values

            self.__case_list.append(case)
    

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
        return self.__name

    @pyqtSlot(result="QVariantList")
    def observationXValues(self):
        return self.__obs_x_values

    @pyqtSlot(result="QVariantList")
    def observationYValues(self):
        return self.__obs_y_values

    @pyqtSlot(result="QVariantList")
    def observationStdValues(self):
        return self.__obs_std_values

    @pyqtSlot(result=bool)
    def observationIsContinuous(self):
        return self.__obs_is_continuous



    @pyqtSlot(result="QVariantList")
    def refcaseXValues(self):
        return self.__refcase_x_values

    @pyqtSlot(result="QVariantList")
    def refcaseYValues(self):
        return self.__refcase_y_values


    @pyqtSlot(QString, result="QVariantList")
    def ensembleXValues(self, case):
        return self.__ensemble_x_values[str(case)]

    @pyqtSlot(QString, result="QVariantList")
    def ensembleYValues(self, case):
        return self.__ensemble_y_values[str(case)]

    @pyqtSlot(QString, result="QVariantList")
    def ensembleMinYValues(self, case):
        return self.__ensemble_y_min_values[str(case)]

    @pyqtSlot(QString, result="QVariantList")
    def ensembleMaxYValues(self, case):
        return self.__ensemble_y_max_values[str(case)]

    @pyqtSlot(QString, result=int)
    def realizationCount(self, case):
        return len(self.__ensemble_y_max_values[str(case)])


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


    @pyqtSlot(result=bool)
    def isValid(self):
        return self.hasBoundaries() and (self.hasObservationData() or self.hasRefcaseData() or self.hasEnsembleData())

    @pyqtSlot(result=bool)
    def hasBoundaries(self):
        return self.__min_x is not None and self.__max_x is not None and self.__min_y is not None and self.__max_y is not None

    @pyqtSlot(result="QStringList")
    def caseList(self):
        return self.__case_list

    @pyqtSlot(result=bool)
    def hasObservationData(self):
        return self.__has_observation_data

    @pyqtSlot(result=bool)
    def hasRefcaseData(self):
        return self.__has_refcase_data

    @pyqtSlot(result=bool)
    def hasEnsembleData(self):
        return len(self.__case_list) > 0

    @pyqtSlot(QString, result="QVariantList")
    def reportStepSamples(self, case_name):
        result = []

        for realization in self.__ensemble_y_values[str(case_name)]:
            result.append(realization[len(realization) - 1])

        return result
    
    



