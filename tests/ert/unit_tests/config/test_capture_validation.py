from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ForwardModelStepWarning,
    capture_validation,
)


def test_capture_validation_captures_warnings():
    with capture_validation() as validation_messages:
        ConfigWarning.warn("Message")

    assert validation_messages.warnings[0].message == "Message"


def test_capture_validation_captures_deprecations():
    with capture_validation() as validation_messages:
        ConfigWarning.deprecation_warn("Message")

    assert validation_messages.deprecations[0].message == "Message"


def test_capture_validation_captures_validation_errors():
    with capture_validation() as validation_messages:
        raise ConfigValidationError("Message")

    assert validation_messages.errors[0].message == "Message"


def test_capture_validation_captures_plugin_warnings():
    with capture_validation() as validation_messages:
        ForwardModelStepWarning.warn("Message")

    assert validation_messages.warnings[0].message == "Message"
