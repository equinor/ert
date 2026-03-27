import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from PyQt6.QtWidgets import QLabel

from ert.ensemble_evaluator.state import REAL_STATE_TO_COLOR
from ert.gui.experiments.view import ProgressWidget
from tests.ert.ui_tests.gui.conftest import get_child


@given(
    status=st.dictionaries(
        st.sampled_from(list(REAL_STATE_TO_COLOR.keys())),
        st.integers(min_value=1, max_value=200),
        min_size=1,
    )
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_marker_label_text_correct(qtbot, status: dict[str, int]):
    realization_count = sum(status.values())
    progress_widget = ProgressWidget()
    qtbot.addWidget(progress_widget)
    progress_widget.update_progress(status, realization_count)

    for state in REAL_STATE_TO_COLOR:
        label_marker = get_child(
            progress_widget,
            QLabel,
            name=f"progress_label_text_{state}",
        )

        assert label_marker
        count = status.get(state, 0)
        assert f" {state} ({count}/{realization_count})" in label_marker.text()


@given(
    status=st.dictionaries(
        st.sampled_from(list(REAL_STATE_TO_COLOR.keys())),
        st.integers(min_value=1, max_value=200),
        min_size=1,
    )
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_progress_state_width_correct(qtbot, status: dict[str, int]):
    realization_count = sum(status.values())
    progress_widget = ProgressWidget()
    qtbot.addWidget(progress_widget)
    status = {"Unknown": realization_count}
    progress_widget.update_progress(status, realization_count)

    progress_marker = get_child(
        progress_widget,
        QLabel,
        name="progress_Unknown",
    )

    assert progress_marker
    base_width = progress_marker.width() / realization_count
    progress_widget.update_progress(status, realization_count)

    for state in REAL_STATE_TO_COLOR:
        progress_marker = get_child(
            progress_widget,
            QLabel,
            name=f"progress_{state}",
        )

        assert progress_marker
        count = status.get(state, 0)
        assert progress_marker.width() == pytest.approx(base_width * count)


def test_progress_state_color_order(qtbot):
    progress_widget = ProgressWidget()
    qtbot.addWidget(progress_widget)
    expected_color_order = [
        "Unknown",
        "Finished",
        "Failed",
        "Running",
        "Pending",
        "Waiting",
    ]

    for i in range(len(expected_color_order)):
        item = progress_widget._horizontal_layout.itemAt(i).widget()
        assert isinstance(item, QLabel)
        assert expected_color_order[i] in item.objectName()
