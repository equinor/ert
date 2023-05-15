from ert.validation import RangeStringArgument


def test_proper_name_argument():
    argument = RangeStringArgument()

    assert argument.validate("1")
    assert argument.validate("1-10")
    assert argument.validate("1-10,11-20")
    assert argument.validate("1-10,11,12,13,14,15,16-20")

    # The empty string is invalid in ERT2. However, it is the only way to
    # specify that all realizations are inactive
    assert not argument.validate("")

    assert not argument.validate("s5")
    assert not argument.validate("1-10,5-4*")

    assert argument.validate("1 - 5, 2,3 ,4")
    assert argument.validate("1 -  5, 2    ,3 ,4")

    argument = RangeStringArgument(max_value=10)

    assert argument.validate("1-5, 9")
    assert not argument.validate("10")
