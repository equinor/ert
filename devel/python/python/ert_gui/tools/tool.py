
class Tool(object):
    def __init__(self, name, icon=None):
        super(Tool, self).__init__()
        self.__icon = icon
        self.__name = name
        self.__parent = None


    def getIcon(self):
        return self.__icon

    def getName(self):
        return self.__name

    def trigger(self):
        raise NotImplementedError()

    def setParent(self, parent):
        self.__parent = parent

    def parent(self):
        return self.__parent



