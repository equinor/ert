import pandas as pd
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QDialog, QTableView, QVBoxLayout, QWidget


class DesignMatrixPanel(QDialog):
    def __init__(
        self,
        design_matrix_df: pd.DataFrame,
        filename: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(f"Design matrix parameters from {filename}")

        table_view = QTableView(self)
        table_view.setEditTriggers(QTableView.NoEditTriggers)

        self.model = self.create_model(design_matrix_df)
        table_view.setModel(self.model)

        table_view.resizeColumnsToContents()
        table_view.resizeRowsToContents()

        layout = QVBoxLayout()
        layout.addWidget(table_view)
        self.setLayout(layout)
        self.adjustSize()

    @staticmethod
    def create_model(design_matrix_df: pd.DataFrame) -> QStandardItemModel:
        model = QStandardItemModel()

        if isinstance(design_matrix_df.columns, pd.MultiIndex):
            header_labels = [str(col[-1]) for col in design_matrix_df.columns]
        else:
            header_labels = design_matrix_df.columns.astype(str).tolist()

        model.setHorizontalHeaderLabels(header_labels)

        for index, _ in design_matrix_df.iterrows():
            items = [
                QStandardItem(str(design_matrix_df.at[index, col]))
                for col in design_matrix_df.columns
            ]
            model.appendRow(items)
        return model
