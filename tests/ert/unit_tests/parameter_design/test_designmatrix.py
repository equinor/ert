from ert.parameter_design import DesignMatrix

from ._configurations import minimal_configuration


def assert_valid_designmatrix(design_values):
    assert design_values.columns[:3].tolist() == ["REAL", "SENSNAME", "SENSCASE"]
    assert design_values["REAL"].tolist() == list(range(len(design_values)))
    assert not design_values.isna().any().any()


def test_that_design_matrix_generates_seed_sensitivity_for_each_repeat():
    config = minimal_configuration()
    repeats = 10
    config.repeats = repeats
    config.rms_seeds = "default"
    config.sensitivities = {
        "rms_seed": {
            "seedname": "RMS_SEED",
            "senstype": "seed",
            "parameters": None,
            "dependencies": {},
        }
    }

    design = DesignMatrix(config)

    assert_valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == repeats
    assert isinstance(design.defaultvalues, dict)
