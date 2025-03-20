from typing import TYPE_CHECKING

import pandas as pd
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets.stringbox import StringBox
from ert.validation import ActiveRange, RangeSubsetStringArgument

if TYPE_CHECKING:
    from ert.config import DesignMatrix


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
        table_view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)

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
        header_labels = design_matrix_df.columns.astype(str).tolist()

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(header_labels)
        for index, _ in design_matrix_df.iterrows():
            items = [
                QStandardItem(str(design_matrix_df.at[index, col]))
                for col in design_matrix_df.columns
            ]
            model.appendRow(items)
        model.setVerticalHeaderLabels(design_matrix_df.index.astype(str).tolist())
        return model

    @staticmethod
    def show_dm_params(design_matrix: "DesignMatrix") -> None:
        viewer = DesignMatrixPanel(
            design_matrix.design_matrix_df,
            design_matrix.xls_filename.name,
        )
        viewer.setMinimumHeight(500)
        viewer.setMinimumWidth(1000)
        viewer.adjustSize()
        viewer.exec()

    @staticmethod
    def get_design_matrix_button(
        active_realizations_field: StringBox, design_matrix: "DesignMatrix"
    ) -> QHBoxLayout:
        active_realizations_field.setValidator(
            RangeSubsetStringArgument(ActiveRange(design_matrix.active_realizations))
        )
        active_realizations_field.model.setValueFromMask(  # type: ignore
            design_matrix.active_realizations
        )
        show_dm_param_button = QPushButton("Show parameters")
        show_dm_param_button.setObjectName("show-dm-parameters")
        show_dm_param_button.setMinimumWidth(50)

        button_layout = QHBoxLayout()
        button_layout.addWidget(show_dm_param_button)
        button_layout.addStretch()  # Add stretch to push the button to the left

        show_dm_param_button.clicked.connect(
            lambda: DesignMatrixPanel.show_dm_params(design_matrix)
        )

        return button_layout
