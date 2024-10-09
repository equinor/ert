import pandas as pd
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QDialog, QTableView, QVBoxLayout


class DesignMatrixPanel(QDialog):
    def __init__(self, design_matrix_df: pd.DataFrame, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Design matrix parameters viewer")

        self.table_view = QTableView(self)
        self.table_view.setEditTriggers(QTableView.NoEditTriggers)

        self.model = self.create_model(design_matrix_df)
        self.table_view.setModel(self.model)

        self.table_view.resizeColumnsToContents()
        self.table_view.resizeRowsToContents()

        # Layout to hold the table view
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)

    @staticmethod
    def create_model(design_matrix_df: pd.DataFrame) -> QStandardItemModel:
        # Create a model
        model = QStandardItemModel()

        if isinstance(design_matrix_df.columns, pd.MultiIndex):
            header_labels = [
                " | ".join(map(str, col)) for col in design_matrix_df.columns
            ]
        else:
            header_labels = design_matrix_df.columns.astype(str).tolist()

        model.setHorizontalHeaderLabels(header_labels)

        # Populate the model with data
        for index, _ in design_matrix_df.iterrows():
            items = [
                QStandardItem(str(design_matrix_df.at[index, col]))
                for col in design_matrix_df.columns
            ]
            model.appendRow(items)
        return model
