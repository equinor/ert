from random import randint

from qtpy.QtWidgets import QLabel

from ert.ensemble_evaluator.state import REAL_STATE_TO_COLOR
from ert.gui.simulation.view import ProgressWidget
from tests.unit_tests.gui.conftest import get_child


def test_progress_step_changing(qtbot):
    progress_widget = ProgressWidget()
    qtbot.addWidget(progress_widget)

    realization_count = 1

    for i in range(len(REAL_STATE_TO_COLOR.keys())):
        status = {}

        # generate new list with one flag set
        for u, state in enumerate(REAL_STATE_TO_COLOR.keys()):
            status[state] = 1 if i == u else 0

        progress_widget.update_progress(status, realization_count)

        for state in REAL_STATE_TO_COLOR:
            label_marker = get_child(
                progress_widget,
                QLabel,
                name=f"progress_label_text_{state}",
            )

            assert label_marker
            count = status[state]
            assert f" {state} ({count}/{realization_count})" in label_marker.text()


def test_progress_state_width_correct(qtbot):
    progress_widget = ProgressWidget()
    qtbot.addWidget(progress_widget)
    status = {"Unknown": 20}
    realization_count = 20
    progress_widget.update_progress(status, realization_count)

    progress_marker = get_child(
        progress_widget,
        QLabel,
        name="progress_Unknown",
    )

    assert progress_marker
    base_width = progress_marker.width() / realization_count

    spread = realization_count
    gen_list = []

    for _ in range(len(REAL_STATE_TO_COLOR) - 1):
        r = randint(0, spread)
        gen_list.append(r)
        spread -= r
    gen_list.append(spread)

    for i, state in enumerate(REAL_STATE_TO_COLOR.keys()):
        status[state] = gen_list[i]

    progress_widget.update_progress(status, realization_count)

    for state in REAL_STATE_TO_COLOR:
        progress_marker = get_child(
            progress_widget,
            QLabel,
            name=f"progress_{state}",
        )

        assert progress_marker
        count = status.get(state, 0)
        assert progress_marker.width() == base_width * count
