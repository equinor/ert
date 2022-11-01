import pytest

from ert._c_wrappers.util.substitution_list import SubstitutionList


def test_substitution_list():
    subst_list = SubstitutionList()

    subst_list.addItem("Key", "Value")

    assert len(subst_list) == 1

    # pylint: disable=pointless-statement
    with pytest.raises(KeyError):
        subst_list[2]
        subst_list["NoSuchKey"]

    assert "Key" in subst_list
    assert subst_list["Key"], "Value"

    subst_list.addItem("Key2", "Value2")
    assert subst_list.keys() == ["Key", "Key2"]

    str_repr = repr(subst_list)
    assert "SubstitutionList" in str_repr
    assert "Key2, Value2" in str_repr
    assert "Key, Value" in str_repr

    assert subst_list.get("nosuchkey", 1729) == 1729
    assert subst_list.get("nosuchkey") is None
    assert subst_list.get(513) is None
    assert dict(subst_list) == {"Key": "Value", "Key2": "Value2"}
