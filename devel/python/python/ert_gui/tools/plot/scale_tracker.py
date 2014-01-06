class ScaleTracker(object):

    def __init__(self, name):
        super(ScaleTracker, self).__init__()

        self.__name = name
        self.__min_scales = {}
        self.__max_scales = {}

    def getMinimumScaleValue(self, key):
        if key in self.__min_scales:
            return self.__min_scales[key]

        return None

    def getMaximumScaleValue(self, key):
        if key in self.__max_scales:
            return self.__max_scales[key]

        return None

    def setScaleValues(self, key, minimum, maximum):
        if minimum is None and key in self.__min_scales:
            del self.__min_scales[key]
        else:
            self.__min_scales[key] = minimum

        if maximum is None and key in self.__max_scales:
            del self.__max_scales[key]
        else:
            self.__max_scales[key] = maximum

