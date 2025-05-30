import warnings

from ert.exceptions._post_simulation_warnings import (
    PostSimulationWarning,
    QtWarningHandler,
)


def test_post_simulation_warnings_picks_up_only_accepted_warning_types():
    warning_handler = QtWarningHandler(
        accepted_categories=[PostSimulationWarning],
        post_simulation_warnings=[],
    )
    warning_to_be_picked_up = "PostSimulationWarning"
    warning_not_to_be_picked_up = "RegularWarning"
    warnings.warn(warning_to_be_picked_up, PostSimulationWarning, stacklevel=2)
    warnings.warn(warning_not_to_be_picked_up, stacklevel=2)

    warning_messages = [
        str(warning) for warning in warning_handler.post_simulation_warnings
    ]
    assert warning_not_to_be_picked_up not in warning_messages
    assert warning_to_be_picked_up in warning_messages
