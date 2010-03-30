from PyQt4 import QtGui, QtCore
import help #todo this is not a nice way of solving this...

def abstract():
    """Abstract keyword that indicate an abstract function"""
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class ContentModel:
    contentModel = None # A hack to have a "static" class variable
    observers = []

    def __init__(self):
        ContentModel.observers.append(self)

    def getFromModel(self):
        return self.getter(ContentModel.contentModel)

    def getter(self, model):
        """Must be implemented to get data from a source."""
        abstract()

    def setter(self, source, value):
        """Must be implemented to update the source with new data."""
        abstract()

    def fetchContent(self):
        """This function is called to tell all inheriting classes to retrieve data from this model"""
        abstract()

    def updateContent(self, value):
        if not ContentModel.contentModel == None :
            self.setter(ContentModel.contentModel, value)

    @classmethod
    def updateObservers(cls):
        for o in ContentModel.observers:
            o.fetchContent()

    @classmethod
    def printObservers(cls):
        for o in ContentModel.observers:
            print o


class HelpedWidget(QtGui.QWidget, ContentModel):
    """
    HelpedWidget is a class that enables embedded help messages in widgets.
    The help button must manually be added to the containing layout with addHelpButton(layout).
    """

    def __init__(self, parent=None, widgetLabel="", helpLabel=""):
        """Creates a widget that can have a help button"""
        QtGui.QWidget.__init__(self, parent)
        ContentModel.__init__(self)

        if not widgetLabel == "":
            self.label = widgetLabel + ":"
        else:
            self.label = ""

        self.helpMessage = help.resolveHelpLabel(helpLabel)
        self.helpLabel = helpLabel

        self.widgetLayout = QtGui.QHBoxLayout()
        #self.setStyleSheet("padding: 2px")
        self.widgetLayout.setMargin(0)
        self.setLayout(self.widgetLayout)

        self.helpButton = QtGui.QToolButton(self)

        self.helpButton.setIcon(QtGui.QIcon.fromTheme("help"))
        self.helpButton.setIconSize(QtCore.QSize(16, 16))
        self.helpButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.helpButton.setAutoRaise(True)

        self.connect(self.helpButton, QtCore.SIGNAL('clicked()'), self.showHelp)

        if self.helpMessage == "":
            self.helpButton.setEnabled(False)

    def getHelpButton(self):
        """Returns the help button or None"""
        try:
          self.helpButton
        except AttributeError:
            self.helpButton = None

        return self.helpButton

    def showHelp(self):
        """Pops up the tooltip associated to the button"""
        QtGui.QToolTip.showText(QtGui.QCursor.pos(), self.helpMessage, self)

    def addHelpButton(self):
        """Adds the help button to the provided layout."""
        if not self.getHelpButton() is None :
            self.addWidget(self.getHelpButton())

    def getLabel(self):
        return self.tr(self.label)

    def addWidget(self, widget):
        self.widgetLayout.addWidget(widget)

    def addStretch(self):
        self.widgetLayout.addStretch(1)