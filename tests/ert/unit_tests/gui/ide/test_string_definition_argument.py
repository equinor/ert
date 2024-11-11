from ert.validation import StringDefinition


def test_validate_success_with_all_required_tokens():
    string_def = StringDefinition(required=["token1", "token2"], invalid=["invalid1"])
    validation_status = string_def.validate("This is a string with token1 and token2")
    assert bool(validation_status) is True
    assert not validation_status.message()


def test_validate_success_with_required_tokens():
    string_def = StringDefinition(required=["token1", "token2"], invalid=["invalid1"])
    validation_status = string_def.validate("This is a string with token1 and token2")
    assert bool(validation_status) is True
    assert not validation_status.message()


def test_validate_failure_with_empty_required_tokens():
    string_def = StringDefinition(optional=False, required=[], invalid=["invalid1"])
    validation_status = string_def.validate("This is a string with invalid1")
    assert bool(validation_status) is False
    assert validation_status.message() == "Contains invalid string invalid1!"


def test_validate_empty_string():
    string_def = StringDefinition(required=["token1"], invalid=["invalid1"])
    validation_status = string_def.validate("")
    assert bool(validation_status) is False
    assert "Missing required token1!" in validation_status.message()
