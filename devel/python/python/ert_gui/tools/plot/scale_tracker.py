class ScaleTracker(object):

    def __init__(self, name):
        super(ScaleTracker, self).__init__()

        self.__name = name
        self.__states = {}

    def getScaleValue(self, key):
        if self.isEnabled(key):
            return self.getState(key)["scale"]
        else:
            return None

    def getState(self, key):
        if key in self.__states:
            return self.__states[key]
        else:
            return {"scale": None, "enabled": False}

    def setScaleValue(self, key, value):
        if self.getState(key) is None:
            self.__states[key] = {"scale": value, "enabled": True}
        else:
            self.getState(key)["scale"] = value

    def isEnabled(self, key):
        return self.getState(key)["enabled"]

    def setEnabled(self, key, enabled):
        if self.getState(key) is None:
            self.__states[key] = {"scale": None, "enabled": enabled}
        else:
            self.getState(key)["enabled"] = enabled


    def setValues(self, key, value, enabled):
        if key not in self.__states:
            self.__states[key] = {"scale": value, "enabled": enabled}
        else:
            self.getState(key)["enabled"] = enabled
            self.getState(key)["scale"] = value