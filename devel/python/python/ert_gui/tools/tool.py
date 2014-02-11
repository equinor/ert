
class Tool(object):
    def __init__(self, name, help_link="", icon=None, enabled=True):
        super(Tool, self).__init__()
        self.__icon = icon
        self.__name = name
        self.__parent = None
        self.__enabled = enabled
        self.__help_link = help_link

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

    def isEnabled(self):
        return self.__enabled

    def getHelpLink(self):
        return self.__help_link
