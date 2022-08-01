import pytest

from res.util.substitution_list import SubstitutionList


def test_substitution_list():
    subst_list = SubstitutionList()

    subst_list.addItem("Key", "Value", "Doc String")

    assert len(subst_list) == 1

    # pylint: disable=pointless-statement
    with pytest.raises(KeyError):
        subst_list[2]
        subst_list["NoSuchKey"]

    with pytest.raises(KeyError):
        subst_list.doc("NoSuchKey")

    assert "Key" in subst_list
    assert subst_list["Key"], "Value"
    assert subst_list.doc("Key"), "Doc String"

    subst_list.addItem("Key2", "Value2", "Doc String2")
    assert subst_list.keys() == ["Key", "Key2"]

    assert "SubstitutionList(len=2)" in repr(subst_list)

    assert subst_list.get("nosuchkey", 1729) == 1729
    assert subst_list.get("nosuchkey") is None
    assert subst_list.get(513) is None
    assert dict(subst_list) == {"Key": "Value", "Key2": "Value2"}
