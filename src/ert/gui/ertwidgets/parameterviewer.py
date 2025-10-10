from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
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
        # Add a horizontal layout for the collapse/expand and update filter buttons
        menu_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Collapse parameters")
        self.toggle_button.clicked.connect(self.toggle_parameters)
        self.parameters_collapsed = False  # Track state
        menu_layout.addWidget(self.toggle_button)

        # Add Update parameter label and checkboxes
        menu_layout.addSpacing(20)
        self.update_label = QLabel("Update parameter")
        menu_layout.addWidget(self.update_label)
        self.checkbox_true = QCheckBox("True")
        self.checkbox_false = QCheckBox("False")
        self.checkbox_true.setChecked(True)
        self.checkbox_false.setChecked(True)
        self.checkbox_true.stateChanged.connect(self.filter_parameters)
        self.checkbox_false.stateChanged.connect(self.filter_parameters)
        menu_layout.addWidget(self.checkbox_true)
        menu_layout.addWidget(self.checkbox_false)

        menu_layout.addStretch()
        main_layout.addLayout(menu_layout)

        self.tree_widget = self._create_parameter_tree()
        main_layout.addWidget(self.tree_widget)
        self.setLayout(main_layout)
        self.resize(400, 500)

    def _create_parameter_tree(self) -> QTreeWidget:
        """Create a tree widget showing parameters grouped by their type."""
        tree = QTreeWidget()
        tree.setHeaderLabel("Parameters")

        self.type_nodes = {}  # Store type nodes for later use
        self.parameter_nodes = []  # Store parameter nodes for filtering
        for parameter in self.merged_parameters:
            if parameter.type not in self.type_nodes:
                self.type_nodes[parameter.type] = QTreeWidgetItem(
                    tree, [parameter.type.upper()]
                )

            parameter_node = QTreeWidgetItem(
                self.type_nodes[parameter.type], [parameter.name]
            )
            parameter_node._ert_update = (
                parameter.update
            )  # Store update value for filtering
            self.parameter_nodes.append(parameter_node)
            QTreeWidgetItem(parameter_node, [f"Update: {parameter.update}"])
            QTreeWidgetItem(parameter_node, [f"Forward Init: {parameter.forward_init}"])
            if isinstance(parameter, GenKwConfig):
                QTreeWidgetItem(parameter_node, [f"Source: {parameter.input_source}"])
                QTreeWidgetItem(
                    parameter_node, [f"Group: {parameter.group_name.upper()}"]
                )

        tree.expandAll()
        return tree

    def toggle_parameters(self):
        if not self.parameters_collapsed:
            # Collapse all parameter nodes (second level), keep type nodes expanded
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

    def filter_parameters(self):
        show_true = self.checkbox_true.isChecked()
        show_false = self.checkbox_false.isChecked()
        for parameter_node in self.parameter_nodes:
            update_val = parameter_node._ert_update
            show = (update_val is True and show_true) or (
                update_val is False and show_false
            )
            parameter_node.setHidden(not show)
        # Optionally, hide type nodes if all children are hidden
        for type_node in self.type_nodes.values():
            any_visible = any(
                not type_node.child(i).isHidden() for i in range(type_node.childCount())
            )
            type_node.setHidden(not any_visible)


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
