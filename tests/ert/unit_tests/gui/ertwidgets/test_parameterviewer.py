import pytest

from ert.config import Field, GenKwConfig, SurfaceConfig
from ert.config.everest_control import EverestControl
from ert.config.field import ErtboxParameters
from ert.config.parameter_config import LocalizationType
from ert.gui.ertwidgets.parameterviewer import (
    ParametersSummary,
    ParametersViewer,
    ParameterTree,
)


def _gen_kw(name, group, update):
    return GenKwConfig(
        name=name,
        group=group,
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=LocalizationType.GLOBAL if update else None,
    )


@pytest.fixture
def parameters():
    return [
        _gen_kw("a", "COEFFS", update=True),
        _gen_kw("b", "COEFFS", update=True),
        _gen_kw("c", "OTHER", update=False),
    ]


def _top_level_labels(tree):
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def test_that_tree_starts_collapsed_with_group_counts(qtbot, parameters):
    tree = ParameterTree()
    qtbot.addWidget(tree)
    tree.set_parameters(parameters)

    assert _top_level_labels(tree) == ["COEFFS (2)", "OTHER (1)"]
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        assert not item.isExpanded()
        # Children are created lazily on first expansion
        assert item.childCount() == 0


def test_that_expanding_creates_parameters_and_details(qtbot, parameters):
    tree = ParameterTree()
    qtbot.addWidget(tree)
    tree.set_parameters(parameters)

    group = tree.topLevelItem(0)
    tree.expandItem(group)
    assert [group.child(i).text(0) for i in range(group.childCount())] == ["a", "b"]

    parameter = group.child(0)
    tree.expandItem(parameter)
    details = [parameter.child(i).text(0) for i in range(parameter.childCount())]
    assert any(d.startswith("Update:") for d in details)
    assert any(d.startswith("Forward Init:") for d in details)
    assert "Group: COEFFS" in details
    assert any(d.startswith("Source:") for d in details)


def test_that_summary_splits_on_updatability_and_counts(qtbot, parameters):
    summary = ParametersSummary()
    qtbot.addWidget(summary)
    summary.set_parameters(parameters)

    assert summary.title.text() == "Parameters (3)"
    assert _top_level_labels(summary.updatable_tree) == ["COEFFS (2)"]
    assert _top_level_labels(summary.non_updatable_tree) == ["OTHER (1)"]


def test_that_summary_keeps_expansion_when_parameters_are_unchanged(qtbot, parameters):
    summary = ParametersSummary()
    qtbot.addWidget(summary)
    summary.set_parameters(parameters)
    item = summary.updatable_tree.topLevelItem(0)
    summary.updatable_tree.expandItem(item)

    summary.set_parameters(list(parameters))

    assert summary.updatable_tree.topLevelItem(0) is item
    assert item.isExpanded()


def test_that_fields_and_surfaces_show_dimensions(qtbot):
    field = Field.model_construct(
        name="PORO",
        ertbox_params=ErtboxParameters(nx=2, ny=3, nz=4),
        update_strategy=None,
        forward_init=False,
    )
    surface = SurfaceConfig.model_construct(
        name="TOP", ncol=5, nrow=6, update_strategy=None, forward_init=False
    )
    tree = ParameterTree()
    qtbot.addWidget(tree)
    tree.set_parameters([field, surface])

    assert _top_level_labels(tree) == ["PORO (2, 3, 4)", "TOP (5, 6)"]


def test_that_other_parameter_types_are_listed_by_name(qtbot):
    control = EverestControl.model_construct(
        name="CONTROL", update_strategy=None, forward_init=False
    )
    tree = ParameterTree()
    qtbot.addWidget(tree)
    tree.set_parameters([control])

    assert _top_level_labels(tree) == ["CONTROL"]


def test_that_the_parameter_viewer_dialog_shows_the_lazy_parameter_trees(
    qtbot, parameters
):
    viewer = ParametersViewer(parameters)
    qtbot.addWidget(viewer)

    assert viewer.parameters_summary.updatable_tree.topLevelItemCount() > 0
