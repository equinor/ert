import os
from PyQt4.QtGui import QWidget, QVBoxLayout
from ert_gui.ide.highlighter import KeywordHighlighter
from ert_gui.ide.ide_panel import IDEPanel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.models.connectors.queue_system.queue_system_selector import QueueSystemSelector
from ert_gui.pages.case_init_configuration import CaseInitializationConfigurationPanel
from ert_gui.pages.queue_system_configuration import QueueSystemConfigurationPanel
from ert_gui.widgets.combo_choice import ComboChoice
from ert_gui.widgets.row_panel import RowPanel
from ert_gui.widgets.search_box import SearchBox


class ConfigurationPanel(RowPanel):

    def __init__(self, config_file):
        RowPanel.__init__(self, "Configuration")

        central_widget = QWidget()

        def getLabel():
            return ""
        central_widget.getLabel = getLabel
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        search = SearchBox()
        ide = IDEPanel()

        layout.addWidget(search)
        layout.addWidget(ide, 1)

        with open(config_file) as f:
            config_file = f.read()

        highlighter = KeywordHighlighter(ide.document())
        ide.handler_names = sorted(highlighter.handler_names)

        search.filterChanged.connect(highlighter.setSearchString)

        ide.document().setPlainText(config_file)

        cursor = ide.textCursor()
        cursor.setPosition(0)
        ide.setTextCursor(cursor)
        ide.setFocus()
        self.addRow(central_widget)

        # self.addLabeledSeparator("Case initialization")
        # case_combo = ComboChoice(CaseSelectorModel(), "Current case", "init/current_case_selection")
        # case_configurator = CaseInitializationConfigurationPanel()
        # self.addRow(case_combo, case_configurator)
        #
        # self.addLabeledSeparator("Queue System")
        #
        # queue_system_selector = QueueSystemSelector()
        # queue_system_combo = ComboChoice(queue_system_selector, "Queue system", "config/queue_system/queue_system")
        # queue_system_configurator = QueueSystemConfigurationPanel()
        # self.addRow(queue_system_combo, queue_system_configurator)




