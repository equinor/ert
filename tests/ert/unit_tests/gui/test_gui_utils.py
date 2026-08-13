import logging

import pytest
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal

from ert.gui.utils import (
    LONGEST_DEFAULT_EXPERIMENT_NAME,
    log_once,
    truncate_dropdown_item,
    truncate_experiment_name,
    truncate_string,
)


class SignalEmitter(QObject):
    emitted = Signal()


@pytest.mark.parametrize(
    ("input_string", "max_length", "expected_output"),
    [
        ("short", 10, "short"),
        ("exactlyten", 10, "exactlyten"),
        (
            "exactlyis11",
            10,
            "exa...is11",
        ),
    ],
)
def test_that_truncate_string_truncates_when_appropriate(
    input_string, max_length, expected_output
):
    truncatedString = truncate_string(input_string, max_length)
    assert truncatedString == expected_output
    if input_string != expected_output:
        assert len(truncatedString) == max_length
    else:
        assert len(truncatedString) == len(input_string)


@pytest.mark.parametrize(
    ("experiment_name", "expected_output"),
    [
        ("ensemble_information_filter", "ensemble_information_filter"),
        ("ensemble_information_filter_11", "ensemble_inf...on_filter_11"),
    ],
)
def test_that_truncate_experiment_name_to_longest_default_length(
    experiment_name, expected_output
):
    truncatedName = truncate_experiment_name(experiment_name)
    assert truncatedName == expected_output
    if experiment_name != expected_output:
        assert len(truncatedName) == LONGEST_DEFAULT_EXPERIMENT_NAME
    else:
        assert len(truncatedName) == len(experiment_name)


@pytest.mark.parametrize(
    ("dropdown_item", "expected_output"),
    [
        ("short item", "short item"),
        ("a" * 100, "a" * 100),
        (
            "a" * 101,
            "a" * 48 + "..." + "a" * 49,
        ),
    ],
)
def test_that_truncate_dropdown_item_truncates_to_100_characters(
    dropdown_item, expected_output
):
    truncatedItem = truncate_dropdown_item(dropdown_item)
    assert truncatedItem == expected_output
    if dropdown_item != expected_output:
        assert len(truncatedItem) == 100
    else:
        assert len(truncatedItem) == len(dropdown_item)


def test_that_log_once_evaluates_message_factory_on_first_signal_emission(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    emitter = SignalEmitter()
    iteration = 0

    log_once(
        emitter.emitted,
        logging.getLogger(__name__),
        lambda: f"Iteration {iteration} selected",
    )

    iteration = 1
    emitter.emitted.emit()
    emitter.emitted.emit()

    assert [record.message for record in caplog.records] == ["Iteration 1 selected"]
