from ert._c_wrappers.enkf.key_manager import KeyManager


def test_summary_keys(snake_oil_case):
    ert = snake_oil_case
    key_man = KeyManager(ert)

    assert len(key_man.summaryKeys()) == 47
    assert "FOPT" in key_man.summaryKeys()

    assert len(key_man.summaryKeysWithObservations()) == 2
    assert "FOPR" in key_man.summaryKeysWithObservations()
    assert key_man.isKeyWithObservations("FOPR")


def test_gen_data_keys(snake_oil_case):
    ert = snake_oil_case
    key_man = KeyManager(ert)

    assert len(key_man.genDataKeys()) == 3
    assert "SNAKE_OIL_WPR_DIFF@199" in key_man.genDataKeys()

    assert len(key_man.genDataKeysWithObservations()) == 1
    assert "SNAKE_OIL_WPR_DIFF@199" in key_man.genDataKeysWithObservations()
    assert key_man.isKeyWithObservations("SNAKE_OIL_WPR_DIFF@199")


def test_gen_kw_keys(snake_oil_case):
    ert = snake_oil_case
    key_man = KeyManager(ert)

    assert len(key_man.genKwKeys()) == 10
    assert "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE" in key_man.genKwKeys()


def test_gen_kw_priors(snake_oil_case):
    ert = snake_oil_case
    key_man = KeyManager(ert)
    priors = key_man.gen_kw_priors()
    assert len(priors["SNAKE_OIL_PARAM"]) == 10
    assert {
        "key": "OP1_PERSISTENCE",
        "function": "UNIFORM",
        "parameters": {"MIN": 0.01, "MAX": 0.4},
    } in priors["SNAKE_OIL_PARAM"]
