from typing import TYPE_CHECKING

import polars as pl
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

if TYPE_CHECKING:
    from ert.config import DesignMatrix


class DesignMatrixPanel(QDialog):
    def __init__(
        self,
        design_matrix_df: pl.DataFrame,
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
    def create_model(design_matrix_df: pl.DataFrame) -> QStandardItemModel:
        header_labels = design_matrix_df.select(pl.exclude("realization")).columns

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(header_labels)
        for row_dict in design_matrix_df.iter_rows(named=True):
            items = [
                QStandardItem(str(row_dict[col]))
                for col in design_matrix_df.select(pl.exclude("realization")).columns
            ]
            model.appendRow(items)
        model.setVerticalHeaderLabels(
            design_matrix_df.get_column("realization").cast(pl.String).to_list()
        )
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
        design_matrix: "DesignMatrix",
        number_of_realizations_label: QLabel,
        config_num_realization: int,
    ) -> QHBoxLayout:
        show_dm_param_button = QPushButton("Show parameters")
        show_dm_param_button.setObjectName("show-dm-parameters")
        show_dm_param_button.setMinimumWidth(50)

        button_layout = QHBoxLayout()
        button_layout.addWidget(show_dm_param_button)
        button_layout.addStretch()  # Add stretch to push the button to the left

        show_dm_param_button.clicked.connect(
            lambda: DesignMatrixPanel.show_dm_params(design_matrix)
        )
        if len(design_matrix.active_realizations) != config_num_realization:
            parent_widget = number_of_realizations_label.parent()

            if isinstance(parent_widget, QWidget) and (
                layout := parent_widget.layout()
            ):
                warning_icon = QLabel()
                warning_icon.setObjectName(
                    "warning_icon_num_realizations_design_matrix"
                )
                style = QApplication.style()
                if style is not None:
                    warning_icon.setPixmap(
                        style.standardIcon(
                            QStyle.StandardPixmap.SP_MessageBoxWarning
                        ).pixmap(16, 16)
                    )
                layout.addWidget(warning_icon)
                layout.setSpacing(2)
                layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

                warning_icon.setToolTip(
                    "Number of realizations was set to "
                    + str(
                        min(
                            config_num_realization,
                            len(design_matrix.active_realizations),
                        )
                    )
                    + " due to different number of realizations in the design matrix "
                    "and NUM_REALIZATIONS in config"
                )
                warning_icon.show()

        return button_layout
