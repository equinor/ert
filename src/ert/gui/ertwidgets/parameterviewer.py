from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.config import GenKwConfig, ParameterConfig


class ParametersViewer(QDialog):
    def __init__(
        self, merged_parameters: list[ParameterConfig], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.merged_parameters = merged_parameters
        self.setWindowTitle("Parameter Viewer")

        main_layout = QVBoxLayout(self)
        self.tree_widget = self._create_parameter_tree()

        main_layout.addWidget(self.tree_widget)
        self.setLayout(main_layout)
        self.resize(400, 400)

    def _create_parameter_tree(self) -> QTreeWidget:
        """Create a tree widget showing parameters grouped by their type."""
        tree = QTreeWidget()
        tree.setHeaderLabel("Parameters")

        type_nodes = {}
        for parameter in self.merged_parameters:
            if parameter.type not in type_nodes:
                type_nodes[parameter.type] = QTreeWidgetItem(
                    tree, [parameter.type.upper()]
                )

            parameter_node = QTreeWidgetItem(
                type_nodes[parameter.type], [parameter.name]
            )
            QTreeWidgetItem(parameter_node, [f"Update: {parameter.update}"])
            QTreeWidgetItem(parameter_node, [f"Forward Init: {parameter.forward_init}"])
            if isinstance(parameter, GenKwConfig):
                QTreeWidgetItem(parameter_node, [f"Source: {parameter.input_source}"])
                QTreeWidgetItem(
                    parameter_node, [f"Group: {parameter.group_name.upper()}"]
                )

        tree.expandAll()
        return tree


def get_parameters_button(
    merged_parameters: list[ParameterConfig], parent: QWidget
) -> QHBoxLayout:
    parameter_viewer_button = QPushButton("Show parameters")
    parameter_viewer_button.setMinimumWidth(50)
    parameter_viewer_button.clicked.connect(
        lambda: _show_parameter_viewer(merged_parameters, parent)
    )

    button_layout = QHBoxLayout()
    button_layout.addWidget(parameter_viewer_button)
    button_layout.addStretch()  # Add stretch to push the button to the left
    return button_layout


def _show_parameter_viewer(
    merged_parameters: list[ParameterConfig], parent: QWidget
) -> None:
    parameter_dialog = ParametersViewer(merged_parameters, parent)
    parameter_dialog.exec()
