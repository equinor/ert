from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.config import ParameterConfig


class ParametersViewer(QDialog):
    def __init__(
        self,
        parameter_configurations: list[ParameterConfig],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._parameter_configurations = parameter_configurations
        self.setWindowTitle("Parameter Viewer")

        main_layout = QVBoxLayout(self)
        # Add a horizontal layout for the collapse/expand and update filter buttons
        menu_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Collapse parameters")
        self.toggle_button.clicked.connect(self.toggle_parameters)
        self.parameters_collapsed = False  # Track state
        menu_layout.addWidget(self.toggle_button, alignment=Qt.AlignmentFlag.AlignLeft)
        menu_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        # Add Update parameter label and dropdown
        self.update_filter_combo = QComboBox()
        self.update_filter_combo.addItems(
            ["All Parameters", "Updatable", "Non-updatable"]
        )
        self.update_filter_combo.currentIndexChanged.connect(self.filter_parameters)
        menu_layout.addWidget(QLabel("Show parameters:"))
        menu_layout.addWidget(
            self.update_filter_combo, alignment=Qt.AlignmentFlag.AlignRight
        )

        main_layout.addLayout(menu_layout)

        self.tree_widget = self._create_parameter_tree()
        main_layout.addWidget(self.tree_widget)
        self.setLayout(main_layout)
        self.resize(400, 500)

    def _create_parameter_tree(self) -> QTreeWidget:
        """Create a tree widget showing parameters grouped by their type."""
        tree = QTreeWidget()
        tree.setHeaderLabel("Parameters")

        self.type_nodes: dict[
            str, QTreeWidgetItem
        ] = {}  # Store type nodes for later use
        self.parameter_nodes = []  # Store parameter nodes for filtering
        for parameter in self._parameter_configurations:
            if parameter.type not in self.type_nodes:
                self.type_nodes[parameter.type] = QTreeWidgetItem(
                    tree, [parameter.type.upper()]
                )

            parameter_node = QTreeWidgetItem(
                self.type_nodes[parameter.type], [parameter.name]
            )
            parameter_node.setData(
                0, Qt.ItemDataRole.UserRole, parameter.update
            )  # Store update value for filtering
            self.parameter_nodes.append(parameter_node)
            QTreeWidgetItem(parameter_node, [f"Update: {parameter.update}"])
            QTreeWidgetItem(parameter_node, [f"Forward Init: {parameter.forward_init}"])
            if parameter.group_name:
                QTreeWidgetItem(
                    parameter_node, [f"Group: {parameter.group_name.upper()}"]
                )
            if hasattr(parameter, "input_source"):
                QTreeWidgetItem(parameter_node, [f"Source: {parameter.input_source}"])

        tree.expandAll()
        return tree

    def toggle_parameters(self) -> None:
        """Toggle collapse/expand details of parameter nodes."""
        if not self.parameters_collapsed:
            for type_node in self.type_nodes.values():
                self.tree_widget.expandItem(type_node)
                for i in range(type_node.childCount()):
                    parameter_node = type_node.child(i)
                    self.tree_widget.collapseItem(parameter_node)
            self.toggle_button.setText("Expand parameters")
            self.parameters_collapsed = True
        else:
            # Expand all parameter nodes (second level), keep type nodes expanded
            for type_node in self.type_nodes.values():
                self.tree_widget.expandItem(type_node)
                for i in range(type_node.childCount()):
                    parameter_node = type_node.child(i)
                    self.tree_widget.expandItem(parameter_node)
            self.toggle_button.setText("Collapse parameters")
            self.parameters_collapsed = False

    def filter_parameters(self) -> None:
        """Filter parameters based on their "updatability"."""
        selected = self.update_filter_combo.currentText()
        for parameter_node in self.parameter_nodes:
            update_val = parameter_node.data(0, Qt.ItemDataRole.UserRole)
            if selected == "All Parameters":
                show = True
            elif selected == "Updatable":
                show = update_val
            else:  # "Non-updatable"
                show = not update_val
            parameter_node.setHidden(not show)


def get_parameters_button(
    parameter_configurations: list[ParameterConfig], parent: QWidget
) -> QHBoxLayout:
    parameter_viewer_button = QPushButton("Show parameters")
    parameter_viewer_button.setMinimumWidth(50)
    parameter_viewer_button.clicked.connect(
        lambda: _show_parameter_viewer(parameter_configurations, parent)
    )

    button_layout = QHBoxLayout()
    button_layout.addWidget(parameter_viewer_button)
    button_layout.addStretch()  # Add stretch to push the button to the left
    return button_layout


def _show_parameter_viewer(
    parameter_configurations: list[ParameterConfig], parent: QWidget
) -> None:
    parameter_dialog = ParametersViewer(parameter_configurations, parent)
    parameter_dialog.exec()
