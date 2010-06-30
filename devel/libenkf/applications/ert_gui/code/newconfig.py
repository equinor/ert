from PyQt4.QtGui import QDialog, QFormLayout, QLabel, QDialogButtonBox, QComboBox, QCheckBox
from PyQt4.QtCore import Qt, SIGNAL
from widgets.util import createSpace
from PyQt4.QtGui import QLineEdit
import os

class NewConfigurationDialog(QDialog):
    """A dialog for selecting defaults for a new configuration."""
    def __init__(self, configuration_filename, parent = None):
        QDialog.__init__(self, parent)

        self.setModal(True)
        self.setWindowTitle("New configuration file")
        self.setMinimumWidth(250)
        self.setMinimumHeight(150)

        layout = QFormLayout()

        directory, filename = os.path.split(configuration_filename)

        if directory.strip() == "":
            directory = os.path.abspath(os.curdir)
            self.configuration_filename = "%s/%s" % (directory, filename)

        configuration_location = QLabel()
        configuration_location.setText(directory)

        configuration_name = QLabel()
        configuration_name.setText(filename)

        self.db_type = QComboBox()
        self.db_type.addItem("PLAIN")
        self.db_type.addItem("BLOCK_FS")

        self.first_case_name = QLineEdit()
        self.first_case_name.setText("default")
        self.connect(self.first_case_name, SIGNAL('textChanged(QString)'), self._validateName)

        layout.addRow(createSpace(10))
        layout.addRow("Configuration name:", configuration_name)
        layout.addRow("Configuration location:", configuration_location)
        layout.addRow("DBase type:", self.db_type)
        layout.addRow("Name of first case:", self.first_case_name)

        layout.addRow(createSpace(10))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)

        layout.addRow(buttons)

        self.connect(buttons, SIGNAL('accepted()'), self.accept)
        self.connect(buttons, SIGNAL('rejected()'), self.reject)


        self.setLayout(layout)

    def getConfigurationFilename(self):
        return self.configuration_filename

    def getCaseName(self):
        """Return the name of the first case."""
        return str(self.first_case_name.text()).strip()

    def getDBaseType(self):
        """Return the DBase type"""
        return str(self.db_type.currentText())

    def _validateName(self, name):
        name = str(name)
        enabled = len(name) > 0 and name.find(" ") == -1
        self.ok_button.setEnabled(enabled)
