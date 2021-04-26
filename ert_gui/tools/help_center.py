from pkg_resources import resource_string


class HelpCenter(object):
    __default_help_string = "No help available!"
    __help_centers = {}

    def __init__(self, name):
        if name in HelpCenter.__help_centers:
            raise UserWarning("HelpCenter '%s' already exists!")

        super(HelpCenter, self).__init__()
        self.__name = name
        self.__listeners = []
        self.__current_help_link = ""
        self.__help_messages = {}

        HelpCenter.__help_centers[name] = self

    def setHelpMessageLink(self, help_link):
        self.__current_help_link = help_link

        help_message = self.resolveHelpLink(help_link)

        if help_message is not None:
            self.__help_messages[help_link] = help_message
        else:
            self.__help_messages[help_link] = self.__default_help_string

        # if not help_link in self.__help_messages:
        #     help_message = self.resolveHelpLink(help_link)
        #     if help_message is not None:
        #         self.__help_messages[help_link] = help_message
        #     else:
        #         self.__help_messages[help_link] = self.__default_help_string

        for listener in self.__listeners:
            listener.setHelpMessage(help_link, self.__help_messages[help_link])

    def addListener(self, listener):
        self.__listeners.append(listener)
        help_link = self.__current_help_link
        listener.setHelpMessage(help_link, self.__help_messages[help_link])

    def getTemplate(self):
        try:
            return resource_string(
                "ert_gui", "resources/gui/help/template.html"
            ).decode("utf-8")
        except IOError:
            return "<html>%s</html>"

    def resolveHelpLink(self, help_link):
        """
        Reads a HTML file from the help directory.
        The HTML must follow the specification allowed by QT here: http://doc.trolltech.com/4.6/richtext-html-subset.html
        """

        # This code can be used to find widgets with empty help labels
        #    if label.strip() == "":
        #        raise AssertionError("NOOOOOOOOOOOOOOOOOOOOO!!!!!!!!!!!!")

        try:
            return self.getTemplate() % resource_string(
                "ert_gui", "resources/gui/help/{}.html".format(help_link)
            ).decode("utf-8")
        except IOError:
            return None

    @classmethod
    def getHelpCenter(cls, name):
        """@rtype: HelpCenter"""
        return HelpCenter.__help_centers.get(name)

    @staticmethod
    def addHelpToAction(action, link, help_center_name="ERT"):
        def showHelp():
            HelpCenter.getHelpCenter(help_center_name).setHelpMessageLink(link)

        action.hovered.connect(showHelp)
