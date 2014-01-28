from PyQt4.QtCore import pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QVBoxLayout, QListView, QColor, QSortFilterProxyModel, QHBoxLayout, QToolButton, \
    QComboBox, QStandardItemModel, QStandardItem
from ert_gui.tools.plot import DataTypeKeysListModel, DataTypeProxyModel
from ert_gui.widgets.legend import Legend
from ert_gui.widgets.search_box import SearchBox


class DataTypeKeysWidget(QWidget):
    dataTypeKeySelected = pyqtSignal(str)

    def __init__(self):
        QWidget.__init__(self)

        layout = QVBoxLayout()

        self.model = DataTypeKeysListModel()
        self.filter_model = DataTypeProxyModel(self.model)

        self.search_box = SearchBox()
        self.search_box.filterChanged.connect(self.setSearchString)

        layout.addWidget(self.search_box)

        data_type_model = QStandardItemModel(0, 1)
        item = QStandardItem("Select data types...")

        data_type_model.appendRow(item)

        self.__summary_item = QStandardItem("Summary")
        self.__summary_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.__summary_item.setData(Qt.Checked, Qt.CheckStateRole)

        data_type_model.appendRow(self.__summary_item)

        self.__block_item = QStandardItem("Block")
        self.__block_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.__block_item.setData(Qt.Checked, Qt.CheckStateRole)
        data_type_model.appendRow(self.__block_item)

        data_type_model.itemChanged.connect(self.onItemChanged)

        combo = QComboBox()
        combo.setModel(data_type_model)
        layout.addWidget(combo)

        self.data_type_keys_widget = QListView()
        self.data_type_keys_widget.setModel(self.filter_model)
        self.data_type_keys_widget.selectionModel().selectionChanged.connect(self.itemSelected)

        layout.addSpacing(15)
        layout.addWidget(self.data_type_keys_widget, 2)

        layout.addWidget(Legend("Default types", DataTypeKeysListModel.DEFAULT_DATA_TYPE))
        layout.addWidget(Legend("Observations available", DataTypeKeysListModel.HAS_OBSERVATIONS))

        self.setLayout(layout)

    def onItemChanged(self, item):
        assert isinstance(item, QStandardItem)
        checked = item.checkState()==Qt.Checked
        if item == self.__block_item:
            self.filter_model.setShowBlockKeys(checked)
        elif item == self.__summary_item:
            self.filter_model.setShowSummaryKeys(checked)


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
