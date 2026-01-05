from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt

from ert.config import ParameterConfig
from ert.gui.ertwidgets.parameterviewer import ParametersViewer


def create_mock_parameter(**kwargs):
    """Helper to create mock parameters with common properties"""
    param = Mock(ParameterConfig)
    for key, value in kwargs.items():
        setattr(param, key, value)
    return param


@pytest.fixture
def parameters():
    """Create mock parameters for all four ParameterConfig subclasses"""
    return [
        create_mock_parameter(
            name="gen_kw_param1",
            type="gen_kw",
            update=True,
            forward_init=False,
            input_source="sampled",
            group_name="groupA",
        ),
        create_mock_parameter(
            name="gen_kw_param2",
            type="gen_kw",
            update=False,
            forward_init=True,
            input_source="design_matrix",
            group_name="groupB",
        ),
        create_mock_parameter(
            name="ext_param1",
            type="everest_parameters",
            update=True,
            forward_init=False,
            group_name="ext_param1",
        ),
        create_mock_parameter(
            name="ext_param2",
            type="everest_parameters",
            update=False,
            forward_init=True,
            group_name="ext_param2",
        ),
        create_mock_parameter(
            name="surface_param1",
            type="surface",
            update=True,
            forward_init=False,
            group_name="surface_param1",
        ),
        create_mock_parameter(
            name="surface_param2",
            type="surface",
            update=False,
            forward_init=True,
            group_name="surface_param2",
        ),
        create_mock_parameter(
            name="field_param1",
            type="field",
            update=True,
            forward_init=False,
            group_name="field_param1",
        ),
        create_mock_parameter(
            name="field_param2",
            type="field",
            update=False,
            forward_init=True,
            group_name="field_param2",
        ),
    ]


@pytest.fixture
def viewer(qtbot, parameters):
    widget = ParametersViewer(parameters)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_that_tree_structure_agrees_with_data(viewer):
    """Test tree contains correct types and parameter names"""
    tree = viewer.tree_widget

    # Check type nodes
    type_names = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
    assert type_names == {"GEN_KW", "EVEREST_PARAMETERS", "SURFACE", "FIELD"}

    # Check each type has 2 parameters with correct names
    expected_params = {
        "GEN_KW": {"gen_kw_param1", "gen_kw_param2"},
        "EVEREST_PARAMETERS": {"ext_param1", "ext_param2"},
        "SURFACE": {"surface_param1", "surface_param2"},
        "FIELD": {"field_param1", "field_param2"},
    }

    for i in range(tree.topLevelItemCount()):
        type_node = tree.topLevelItem(i)
        type_name = type_node.text(0)
        assert type_node.childCount() == 2

        param_names = {
            type_node.child(j).text(0) for j in range(type_node.childCount())
        }
        assert param_names == expected_params[type_name]


def test_that_parameter_details_agrees_with_data(viewer):
    """Test parameters show extra details as expected"""
    tree = viewer.tree_widget

    for i in range(tree.topLevelItemCount()):
        type_node = tree.topLevelItem(i)

        for j in range(type_node.childCount()):
            node = type_node.child(j)
            assert node is not None

            node_details = [node.child(k).text(0) for k in range(node.childCount())]
            assert any("Update:" in text for text in node_details)
            assert any("Forward Init:" in text for text in node_details)
            assert any("Group:" in text for text in node_details)

            if type_node.text(0) == "GEN_KW":
                assert node.childCount() == 4  # Update, Forward Init, Source, Group
                assert any("Source:" in text for text in node_details)
            else:
                assert node.childCount() == 3
                assert not any("Source:" in text for text in node_details)


def test_that_collapse_expand_happens_when_button_clicked(viewer, qtbot):
    """Test collapse/expand button functionality"""
    tree = viewer.tree_widget

    def all_params_expanded():
        return all(
            tree.isExpanded(tree.indexFromItem(tree.topLevelItem(i).child(j)))
            for i in range(tree.topLevelItemCount())
            for j in range(tree.topLevelItem(i).childCount())
        )

    # Initially expanded
    assert all_params_expanded()

    # Click to collapse
    qtbot.mouseClick(viewer.toggle_button, Qt.MouseButton.LeftButton)
    assert not all_params_expanded()
    assert viewer.toggle_button.text() == "Expand parameters"

    # Click to expand
    qtbot.mouseClick(viewer.toggle_button, Qt.MouseButton.LeftButton)
    assert all_params_expanded()
    assert viewer.toggle_button.text() == "Collapse parameters"


@pytest.mark.parametrize(
    ("filter_value", "expected_update_values"),
    [
        ("Updatable", [True]),
        ("Non-updatable", [False]),
        ("All Parameters", [True, False]),
    ],
)
def test_that_filtering_shows_expected_nodes(
    viewer, filter_value, expected_update_values
):
    """Test filtering dropdown functionality"""
    tree = viewer.tree_widget
    combo = viewer.update_filter_combo

    combo.setCurrentText(filter_value)
    viewer.filter_parameters()

    # Check visible parameters have expected update values
    for i in range(tree.topLevelItemCount()):
        type_node = tree.topLevelItem(i)
        for j in range(type_node.childCount()):
            param_node = type_node.child(j)
            update_val = param_node.data(0, Qt.ItemDataRole.UserRole)

            if not param_node.isHidden():
                assert update_val in expected_update_values
