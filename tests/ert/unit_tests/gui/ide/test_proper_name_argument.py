from ert.validation import ProperNameArgument


def test_proper_name_argument():
    argument = ProperNameArgument()

    assert argument.validate("NAME")
    assert argument.validate("__NAME__")
    assert argument.validate("<NAME>")
    assert argument.validate("-NAME-")

    assert not argument.validate("-NA ME-")
    assert not argument.validate("NAME*")
