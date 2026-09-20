import warnings

import pytest

from ert.warnings import capture_specific_warning


@pytest.mark.filterwarnings("ignore:Bar")
def test_capture_specific_warning_propagates_specific_warning():
    class SpecificWarning(Warning):
        pass

    warning_to_be_captured = "Foo"
    warning_not_to_be_captured = "Bar"

    specific_warnings = []
    warning_propagation_method = specific_warnings.append

    with capture_specific_warning(SpecificWarning, warning_propagation_method):
        warnings.warn(warning_to_be_captured, SpecificWarning, stacklevel=2)
        warnings.warn(warning_not_to_be_captured, UserWarning, stacklevel=2)

    warning_messages = [str(warning_event) for warning_event in specific_warnings]
    assert warning_to_be_captured in warning_messages
    assert warning_not_to_be_captured not in warning_messages
