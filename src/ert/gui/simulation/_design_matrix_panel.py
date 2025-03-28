from typing import TYPE_CHECKING

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets.stringbox import StringBox
from ert.validation import ActiveRange, RangeSubsetStringArgument

from ..ertwidgets.models import ActiveRealizationsModel

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
        active_realizations_field: StringBox,
        design_matrix: "DesignMatrix",
        ensemble_size_label: QLabel,
        ensemble_size: int,
    ) -> QHBoxLayout:
        active_realizations_field.setValidator(
            RangeSubsetStringArgument(ActiveRange(design_matrix.active_realizations))
        )
        assert isinstance(active_realizations_field.model, ActiveRealizationsModel)
        active_realizations_field.model.ensemble_size = len(
            design_matrix.active_realizations
        )
        active_realizations_field.model.setValueFromMask(
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
        dm_ensemble_size = sum(design_matrix.active_realizations)
        if dm_ensemble_size != ensemble_size:
            ensemble_size_label.setText(f"<b>{dm_ensemble_size}</b>")
            parent_widget = ensemble_size_label.parent()

            if isinstance(parent_widget, QWidget) and parent_widget.layout():
                warning_icon = QLabel()
                style = QApplication.style()
                if style is not None:
                    warning_icon.setPixmap(
                        style.standardIcon(
                            QStyle.StandardPixmap.SP_MessageBoxWarning
                        ).pixmap(16, 16)
                    )
                layeout = parent_widget.layout()
                if layeout is not None:
                    layeout.addWidget(warning_icon)
                    layeout.setSpacing(2)
                    layeout.setAlignment(Qt.AlignmentFlag.AlignLeft)

                warning_icon.setToolTip(
                    f"Ensemble size changed from {ensemble_size} due to Design Matrix entries"
                )
                warning_icon.show()

        return button_layout
