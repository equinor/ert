from ecl.util.util import BoolVector
from ert_gui.ertwidgets.models.valuemodel import ValueModel
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount


def mask_to_rangestring(mask):
    """Convert a mask (ordered collection of booleans) into a range string

    For instance, `0 1 0 1 1 1` would be converted to `1, 3-5`
    """
    ranges = []

    def store_range(begin, end):
        if end - begin == 1:
            ranges.append("{}".format(begin))
        else:
            ranges.append("{}-{}".format(begin, end - 1))

    start = None
    for i, is_active in enumerate(mask):
        if is_active:
            if start is None:  # begin tracking a range
                start = i
            assert start is not None
        else:
            if start is not None:  # store the range and stop tracking
                store_range(start, i)
                start = None
            assert start is None
    if start is not None:  # complete the last range if any
        store_range(start, len(mask))
    return ",".join(ranges)


class ActiveRealizationsModel(ValueModel):
    def __init__(self):
        ValueModel.__init__(self, self.getDefaultValue())
        self._custom = False

    def setValue(self, active_realizations):
        if (
            active_realizations is None
            or active_realizations.strip() == ""
            or active_realizations == self.getDefaultValue()
        ):
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, active_realizations)

    def setValueFromMask(self, mask):
        self.setValue(mask_to_rangestring(mask))

    def getDefaultValue(self):
        size = getRealizationCount()
        return "0-%d" % (size - 1)

    def getActiveRealizationsMask(self):
        count = getRealizationCount()

        mask = BoolVector(default_value=False, initial_size=count)
        if not mask.updateActiveMask(self.getValue()):
            raise ValueError('Error while parsing range string "%s"!' % self.getValue())

        if len(mask) != count:
            raise ValueError("Mask size changed %d != %d!" % (count, len(mask)))

        return mask
