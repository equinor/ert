from PyQt4.QtCore import pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QVBoxLayout, QListView, QColor, QSortFilterProxyModel
from ert_gui.tools.plot import DataTypeKeysListModel
from ert_gui.widgets.legend import Legend
from ert_gui.widgets.search_box import SearchBox


class DataTypeKeysWidget(QWidget):
    dataTypeKeySelected = pyqtSignal(str)

    def __init__(self):
        QWidget.__init__(self)

        layout = QVBoxLayout()

        self.model = DataTypeKeysListModel()
        self.filter_model = QSortFilterProxyModel()
        self.filter_model.setSourceModel(self.model)
        self.filter_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.search_box = SearchBox()
        self.search_box.filterChanged.connect(self.setSearchString)
        layout.addWidget(self.search_box)



        self.data_type_keys_widget = QListView()
        self.data_type_keys_widget.setModel(self.filter_model)
        self.data_type_keys_widget.selectionModel().selectionChanged.connect(self.itemSelected)

        layout.addWidget(self.data_type_keys_widget, 2)

        layout.addWidget(Legend("Default types", DataTypeKeysListModel.DEFAULT_DATA_TYPE))
        layout.addWidget(Legend("Observations available", DataTypeKeysListModel.HAS_OBSERVATIONS))

        self.setLayout(layout)


    def itemSelected(self):
        selected_item = self.getSelectedItem()
        if selected_item is not None:
            self.dataTypeKeySelected.emit(selected_item)


    def getSelectedItem(self):
        """ @rtype: str """
        index = self.data_type_keys_widget.currentIndex()
        source_index = self.filter_model.mapToSource(index)
        item = self.model.itemAt(source_index)
        return item

    def selectDefault(self):
        self.data_type_keys_widget.setCurrentIndex(self.filter_model.index(0, 0))


    def setSearchString(self, filter):
        self.filter_model.setFilterFixedString(filter)
