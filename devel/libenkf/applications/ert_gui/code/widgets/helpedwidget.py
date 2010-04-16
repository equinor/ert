from PyQt4 import QtGui, QtCore
import help #todo this is not a nice way of solving this...
import sys
from widgets.util import resourceIcon
import inspect

def abstract():
    """Abstract keyword that indicate an abstract function"""
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class ContentModel:
    """This class is a wrapper for communication between the model and the view."""
    contentModel = None # A hack to have a "static" class variable
    signalManager = QtCore.QObject()
    observers = []

    def __init__(self):
        """Constructs a ContentModel. All inheritors are registered"""
        ContentModel.observers.append(self)

    def initialize(self, model):
        """Should be implemented by inheriting classes that wants to do some one time initialization before getting and setting."""
        abstract()

    def getter(self, model):
        """MUST be implemented to get data from a source. Should not be called directly."""
        abstract()

    def setter(self, model, value):
        """MUST be implemented to update the source with new data. Should not be called directly."""
        abstract()

    def fetchContent(self):
        """MUST be implemented. This function is called to tell all inheriting classes to retrieve data from the model. """
        abstract()

    def getFromModel(self):
        """Retrieves the data from the model. Calls the getter function with an appropriate model."""
        gargs = inspect.getargspec(self.getter)

        if inspect.isfunction(self.getter) and "self" in gargs[0]:
            return self.getter.__call__(self, ContentModel.contentModel)
        else:
            return self.getter(ContentModel.contentModel)

    def updateContent(self, value):
        """
        Sends updated data to the model.
        Calls the setter function with an appropriate model.
        Emits a SIGNAL 'contentsChanged()' after the setter has been called.
        """
        if not ContentModel.contentModel is None :
            self.setter(ContentModel.contentModel, value)
            self.emit(QtCore.SIGNAL('contentsChanged()'))

    def getModel(self):
        """Returns the contentModel associated with this session"""
        return ContentModel.contentModel
        
    # The modelConnect, modelDisconnect and modelEmit uses Qt signal handling to enable communication between
    # separate parts of the model that needs to know about changes.
    #todo: can be static?
    def modelConnect(self, signal, callable):
        """Connect to a custom signal available to all ContentModel objects."""
        QtCore.QObject.connect(ContentModel.signalManager, QtCore.SIGNAL(signal), callable)

    def modelDisconnect(self, signal, callable):
        """Disconnect from a custom signal available to all ContentModel objects."""
        QtCore.QObject.disconnect(ContentModel.signalManager, QtCore.SIGNAL(signal), callable)

    def modelEmit(self, signal, *args):
        """Emit a custom signal available to all ContentModel objects."""
        ContentModel.signalManager.emit(QtCore.SIGNAL(signal), *args)

    @classmethod
    def updateObservers(cls):
        """Calls all ContentModel inheritors to initialize (if implemented) and perform initial fetch of data."""
        for o in ContentModel.observers:
            try:
                o.initialize(ContentModel.contentModel)
            except NotImplementedError:
                sys.stderr.write("Missing initializer: " + o.helpLabel + "\n")

            o.fetchContent()

    @classmethod
    def printObservers(cls):
        """Convenience method for printing the registered inheritors."""
        for o in ContentModel.observers:
            print o


class HelpedWidget(QtGui.QWidget, ContentModel):
    """
    HelpedWidget is a class that enables embedded help messages in widgets.
    The help button must manually be added to the containing layout with addHelpButton().
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

        self.helpButton = QtGui.QToolButton(self)

        self.helpButton.setIcon(resourceIcon("help"))
        self.helpButton.setIconSize(QtCore.QSize(16, 16))
        self.helpButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.helpButton.setAutoRaise(True)

        self.connect(self.helpButton, QtCore.SIGNAL('clicked()'), self.showHelp)

        if self.helpMessage == "":
            self.helpButton.setEnabled(False)

        if not self.getHelpButton() is None :
            self.addWidget(self.getHelpButton())

    def getLabel(self):
        """Returns the label of this widget if set or empty string."""
        return self.tr(self.label)

    def addLayout(self, layout):
        """Add a layout to the layout of this widget."""
        self.widgetLayout.addLayout(layout)


    def addWidget(self, widget):
        """Add a widget to the layout of this widget."""
        self.widgetLayout.addWidget(widget)

    def addStretch(self):
        """Add stretch between widgets. Usually added between a widget and the help button."""
        self.widgetLayout.addStretch(1)