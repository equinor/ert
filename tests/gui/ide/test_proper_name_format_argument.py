from ert.validation import ProperNameFormatArgument


def test_proper_name_format_argument():
    argument = ProperNameFormatArgument()

    assert argument.validate("NAME%d")
    assert argument.validate("__NA%dME__")
    assert argument.validate("<NAME>%d")
    assert argument.validate("%d-NAME-")

    assert not argument.validate("-%dNA ME-")
    assert not argument.validate("NAME*%d")
