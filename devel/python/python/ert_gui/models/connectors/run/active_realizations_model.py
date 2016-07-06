from ert.util import BoolVector
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import BasicModelMixin


class ActiveRealizationsModel(ErtConnector, BasicModelMixin):


    def __init__(self):
        self.__active_realizations = self.getDefaultValue()
        self.__custom = False
        super(ActiveRealizationsModel, self).__init__()


    def getValue(self):
        """ @rtype: str """
        return self.__active_realizations

    def setValue(self, active_realizations):
        if active_realizations is None or active_realizations.strip() == "" or active_realizations == self.getDefaultValue():
            self.__custom = False
            self.__active_realizations = self.getDefaultValue()
        else:
            self.__custom = True
            self.__active_realizations = active_realizations

        self.observable().notify(self.VALUE_CHANGED_EVENT)


    def getDefaultValue(self):
        size = getRealizationCount()
        return "0-%d" % (size - 1)

    def getActiveRealizationsMask(self):
        count = getRealizationCount()

        mask = BoolVector.createActiveMask(self.getValue())

        if mask is None:
            raise ValueError("Error while parsing range string!")

        if len(mask) > count:
            raise ValueError("Mask size changed %d != %d!" % (count, len(mask)))

        return mask












