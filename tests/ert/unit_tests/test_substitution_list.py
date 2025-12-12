from ert.config import ErtConfig
from ert.config.parsing import ConfigKeys
from ert.substitutions import Substitutions


def test_that_define_and_data_kw_parameters_are_used_as_substitutions():
    substitutions = ErtConfig.from_dict(
        {
            ConfigKeys.NUM_REALIZATIONS: 1,
            ConfigKeys.DEFINE: [
                ("keyA", "valA"),
                ("keyB", "valB"),
            ],
            ConfigKeys.DATA_KW: [("keyC", "valC"), ("keyD", "valD")],
            ConfigKeys.ENSPATH: "test",
        }
    ).substitutions
    assert substitutions["keyA"] == "valA"
    assert substitutions["keyB"] == "valB"
    assert substitutions["keyC"] == "valC"
    assert substitutions["keyD"] == "valD"


def test_that_delitem_will_remove_substitution():
    substitutions = Substitutions({"<keyA>": "valA", "<keyB>": "valB"})
    assert substitutions.substitute("<keyA><keyB>") == "valAvalB"
    del substitutions["<keyA>"]
    assert substitutions.substitute("<keyA><keyB>") == "<keyA>valB"


def test_that_setitem_will_add_substitution():
    substitutions = Substitutions({"<keyA>": "valA", "<keyB>": "valB"})
    assert substitutions.substitute("<keyA><keyB><keyC>") == "valAvalB<keyC>"
    substitutions["<keyC>"] = "valC"
    assert substitutions.substitute("<keyA><keyB><keyC>") == "valAvalBvalC"
