from ert.shared.ide.keywords.definitions import IntegerArgument


def test_default_integer_argument():
    integer = IntegerArgument()

    validation_status = integer.validate("45")

    assert validation_status
    assert validation_status.value() == "45"
    assert validation_status.message() == ""

    validation_status = integer.validate("-45")

    assert validation_status
    assert validation_status.value() == "-45"

    validation_status = integer.validate("45 ")

    assert not validation_status
    assert validation_status.message() != ""
    assert validation_status.value() is None

    validation_status = integer.validate("gx")

    assert not validation_status
    assert validation_status.message() != ""


def test_integer_range_argument_from():
    from_value = 99
    integer = IntegerArgument(from_value=from_value)

    assert integer.validate(f"{from_value}")

    value = 98
    validation_status = integer.validate(f"{value}")
    assert not validation_status

    assert (
        validation_status.message()
        == IntegerArgument.NOT_IN_RANGE % f"{from_value} <= {value}"
    )


def test_integer_range_argument_to():
    to_value = 99
    integer = IntegerArgument(to_value=to_value)

    assert integer.validate(f"{to_value}")

    value = 100
    validation_status = integer.validate(f"{value}")
    assert not validation_status

    assert (
        validation_status.message()
        == IntegerArgument.NOT_IN_RANGE % f"{value} <= {to_value}"
    )


def test_integer_range_argument():
    from_value = 10
    to_value = 20
    integer = IntegerArgument(from_value=from_value, to_value=to_value)

    assert integer.validate(f"{to_value}")

    assert integer.validate(f"{from_value}")

    assert integer.validate("15")

    value = 9
    validation_status = integer.validate(f"{value}")
    assert not validation_status

    assert (
        validation_status.message()
        == IntegerArgument.NOT_IN_RANGE % f"{from_value} <= {value} <= {to_value}"
    )

    value = 21
    validation_status = integer.validate(f"{value}")
    assert not validation_status

    assert (
        validation_status.message()
        == IntegerArgument.NOT_IN_RANGE % f"{from_value} <= {value} <= {to_value}"
    )
