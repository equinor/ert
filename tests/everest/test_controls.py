import pytest
from pydantic import ValidationError

from ert.config import ConfigWarning
from everest.config import EverestConfig, InputConstraintConfig
from everest.config.control_config import ControlConfig
from everest.config.control_variable_config import (
    ControlVariableConfig,
)
from everest.config.well_config import WellConfig
from everest.optimizer.everest2ropt import everest2ropt


def test_that_duplicate_control_group_name_is_invalid(min_config):
    existing_name = min_config["controls"][0]["name"]

    min_config["controls"].append(
        {
            "name": existing_name,
            "type": "generic_control",
            "variables": [{"name": "var_b", "min": 0, "max": 1, "initial_guess": 0.9}],
        }
    )

    with pytest.raises(ValidationError, match=r"Subfield\(s\) `name` must be unique"):
        EverestConfig.model_validate(min_config)


def test_that_duplicate_control_group_names_without_index_is_invalid():
    with pytest.raises(
        ValidationError, match=r"Subfield\(s\) `name.index` must be unique"
    ):
        ControlConfig(
            name="group",
            type="generic_control",
            initial_guess=0.5,
            variables=[
                ControlVariableConfig(name="w00", min=0, max=1),
                ControlVariableConfig(
                    name="w00", min=0, max=1
                ),  # This is the duplicate
            ],
        )


def test_that_partial_use_of_index_in_control_variables_is_invalid():
    with pytest.raises(
        ValidationError, match="for all of the variables or for none of them"
    ):
        ControlConfig(
            name="group",
            type="generic_control",
            initial_guess=0.5,
            variables=[
                ControlVariableConfig(name="w00", min=0, max=1),
                ControlVariableConfig(name="w01", min=0, max=1, index=0),
            ],
        )


def test_that_duplicate_control_variable_name_and_index_is_invalid():
    with pytest.raises(
        ValidationError, match=r"Subfield\(s\) `name.index` must be unique"
    ):
        ControlConfig(
            name="group",
            type="generic_control",
            initial_guess=0.5,
            variables=[
                ControlVariableConfig(name="w00", min=0, max=1, index=0),
                ControlVariableConfig(name="w00", min=0, max=1, index=0),
            ],
        )


def test_that_unmatched_weight_name_due_to_missing_index_is_invalid():
    with pytest.raises(
        ValidationError,
        match="does not match any instance of "
        "control_name\\.variable_name\\.variable_index",
    ):
        EverestConfig.with_defaults(
            controls=[
                ControlConfig(
                    name="group",
                    type="generic_control",
                    initial_guess=0.5,
                    variables=[
                        ControlVariableConfig(name="w00", min=0, max=1, index=0),
                        ControlVariableConfig(name="w01", min=0, max=1, index=0),
                    ],
                )
            ],
            input_constraints=[
                InputConstraintConfig(
                    upper_bound=1,
                    lower_bound=0,
                    weights={"group.w00": 0.1},
                )
            ],
        )


def test_that_input_constraint_with_deprecated_indexed_name_format_warns():
    with pytest.warns(
        ConfigWarning, match="Deprecated input control name: group.w00-0"
    ):
        EverestConfig.with_defaults(
            controls=[
                ControlConfig(
                    name="group",
                    type="generic_control",
                    initial_guess=0.5,
                    variables=[
                        ControlVariableConfig(name="w00", min=0, max=1, index=0),
                        ControlVariableConfig(name="w01", min=0, max=1, index=0),
                    ],
                )
            ],
            input_constraints=[
                InputConstraintConfig(
                    upper_bound=1,
                    lower_bound=0,
                    weights={
                        "group.w00-0": 0.1
                    },  # This specific format is deprecated [7].
                )
            ],
        )


def test_that_control_variable_with_initial_guess_below_min_is_invalid():
    with pytest.raises(ValidationError, match="initial_guess"):
        ControlConfig(
            name="my_control",
            type="well_control",
            variables=[
                ControlVariableConfig(name="w00", min=0.5, max=1.0, initial_guess=0.3)
            ],
        )


def test_that_control_variable_with_initial_guess_above_max_is_invalid():
    with pytest.raises(ValidationError, match="initial_guess"):
        ControlConfig(
            name="my_control",
            type="well_control",
            variables=[
                ControlVariableConfig(name="w00", min=0.5, max=1.0, initial_guess=1.3)
            ],
        )


def test_that_control_variable_name_with_too_many_dots_is_invalid(min_config):
    illegal_name = "illegal.name.due.to.dots"
    min_config["controls"][0]["variables"][0]["name"] = illegal_name
    with pytest.raises(
        ValidationError,
        match="Variable name can not contain any dots",
    ):
        ControlConfig.model_validate(min_config["controls"][0])


def test_that_control_variable_without_too_many_dots_does_not_raise(min_config):
    weirdo_name = "something/with-symbols_=/()*&%$#!"
    new_config = EverestConfig.model_validate(min_config)
    new_config.wells.append(WellConfig(name=weirdo_name))
    new_config.controls[0].variables[0].name = weirdo_name
    EverestConfig.model_validate(new_config.to_dict())


def test_that_control_variables_not_matching_any_well_name_is_invalid(min_config):
    illegal_name = "nowell4sure"
    min_config["controls"][0]["variables"][0]["name"] = illegal_name
    min_config["controls"][0]["type"] = "well_control"
    with pytest.raises(
        ValidationError,
        match="Variable name does not match any well name",
    ):
        EverestConfig.with_defaults(**(min_config | {"wells": [{"name": "a"}]}))


def test_that_controls_ordering_is_the_same_for_ropt_and_extparam():
    index_wise = ControlConfig(
        name="well_priorities",
        type="well_control",
        variables=[
            {"name": "WELL-1", "initial_guess": [0.58, 0.54, 0.5, 0.52]},
            {"name": "WELL-2", "initial_guess": [0.5, 0.58, 0.56, 0.54]},
            {"name": "WELL-3", "initial_guess": [0.56, 0.52, 0.58, 0.5]},
            {"name": "WELL-4", "initial_guess": [0.54, 0.56, 0.54, 0.58]},
            {"name": "WELL-5", "initial_guess": [0.52, 0.5, 0.52, 0.56]},
        ],
        control_type="real",
        min=0.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.05,
        scaled_range=[0.0, 1.0],
    )

    var_wise = ControlConfig(
        name="well_priorities",
        type="well_control",
        variables=[
            {"name": "WELL-1", "initial_guess": 0.58, "index": 1},
            {"name": "WELL-1", "initial_guess": 0.54, "index": 2},
            {"name": "WELL-1", "initial_guess": 0.5, "index": 3},
            {"name": "WELL-1", "initial_guess": 0.52, "index": 4},
            {"name": "WELL-2", "initial_guess": 0.5, "index": 1},
            {"name": "WELL-2", "initial_guess": 0.58, "index": 2},
            {"name": "WELL-2", "initial_guess": 0.56, "index": 3},
            {"name": "WELL-2", "initial_guess": 0.54, "index": 4},
            {"name": "WELL-3", "initial_guess": 0.56, "index": 1},
            {"name": "WELL-3", "initial_guess": 0.52, "index": 2},
            {"name": "WELL-3", "initial_guess": 0.58, "index": 3},
            {"name": "WELL-3", "initial_guess": 0.5, "index": 4},
            {"name": "WELL-4", "initial_guess": 0.54, "index": 1},
            {"name": "WELL-4", "initial_guess": 0.56, "index": 2},
            {"name": "WELL-4", "initial_guess": 0.54, "index": 3},
            {"name": "WELL-4", "initial_guess": 0.58, "index": 4},
            {"name": "WELL-5", "initial_guess": 0.52, "index": 1},
            {"name": "WELL-5", "initial_guess": 0.5, "index": 2},
            {"name": "WELL-5", "initial_guess": 0.52, "index": 3},
            {"name": "WELL-5", "initial_guess": 0.56, "index": 4},
        ],
        control_type="real",
        min=0.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.05,
        scaled_range=(0.0, 1.0),
    )

    ever_config_var_wise = EverestConfig.with_defaults(controls=[var_wise])
    ever_config_index_wise = EverestConfig.with_defaults(controls=[index_wise])

    ropt_var_wise = everest2ropt(
        ever_config_var_wise.controls,
        ever_config_var_wise.objective_functions,
        ever_config_var_wise.input_constraints,
        ever_config_var_wise.output_constraints,
        ever_config_var_wise.optimization,
        ever_config_var_wise.model,
        1234,
        "dummy",
    )

    ropt_index_wise = everest2ropt(
        ever_config_index_wise.controls,
        ever_config_index_wise.objective_functions,
        ever_config_index_wise.input_constraints,
        ever_config_index_wise.output_constraints,
        ever_config_index_wise.optimization,
        ever_config_index_wise.model,
        1234,
        "dummy",
    )

    assert (
        ropt_var_wise[0]["names"]["variable"] == ropt_index_wise[0]["names"]["variable"]
    )

    assert (
        ropt_var_wise[0]["names"]["variable"]
        == index_wise.to_ert_parameter_config().input_keys
    )

    assert (
        index_wise.to_ert_parameter_config().input_keys
        == var_wise.to_ert_parameter_config().input_keys
    )


def test_that_controls_ordering_disregards_index():
    var_wise = ControlConfig(
        name="well_priorities",
        type="well_control",
        variables=[
            {"name": "WELL-1", "initial_guess": 0.54, "index": 2},
            {"name": "WELL-1", "initial_guess": 0.58, "index": 1},
            {"name": "WELL-1", "initial_guess": 0.5, "index": 3},
            {"name": "WELL-2", "initial_guess": 0.58, "index": 2},
            {"name": "WELL-2", "initial_guess": 0.5, "index": 1},
            {"name": "WELL-2", "initial_guess": 0.56, "index": 3},
            {"name": "WELL-3", "initial_guess": 0.52, "index": 2},
            {"name": "WELL-3", "initial_guess": 0.56, "index": 1},
            {"name": "WELL-3", "initial_guess": 0.58, "index": 3},
        ],
        control_type="real",
        min=0.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.05,
        scaled_range=(0.0, 1.0),
    )

    ever_config_var_wise = EverestConfig.with_defaults(controls=[var_wise])

    ropt_var_wise = everest2ropt(
        ever_config_var_wise.controls,
        ever_config_var_wise.objective_functions,
        ever_config_var_wise.input_constraints,
        ever_config_var_wise.output_constraints,
        ever_config_var_wise.optimization,
        ever_config_var_wise.model,
        1234,
        "dummy",
    )

    expected = [
        "well_priorities.WELL-1.2",
        "well_priorities.WELL-1.1",
        "well_priorities.WELL-1.3",
        "well_priorities.WELL-2.2",
        "well_priorities.WELL-2.1",
        "well_priorities.WELL-2.3",
        "well_priorities.WELL-3.2",
        "well_priorities.WELL-3.1",
        "well_priorities.WELL-3.3",
    ]
    assert (ropt_var_wise[0]["names"]["variable"]) == expected

    assert var_wise.to_ert_parameter_config().input_keys == expected


def test_that_setting_initial_guess_in_a_list_is_the_same_as_one_per_index():
    controls1 = ControlConfig(
        name="controls",
        type="generic_control",
        variables=[
            {"name": "var", "initial_guess": [0.1, 0.2, 0.3]},
        ],
        control_type="real",
        min=0.0,
        max=1.0,
        perturbation_type="relative",
        perturbation_magnitude=5,
        scaled_range=[1.0, 2.0],
        enabled=False,
    )

    controls2 = ControlConfig(
        name="controls",
        type="generic_control",
        variables=[
            {"name": "var", "initial_guess": 0.1, "index": 1},
            {"name": "var", "initial_guess": 0.2, "index": 2},
            {"name": "var", "initial_guess": 0.3, "index": 3},
        ],
        control_type="real",
        min=0.0,
        max=1.0,
        perturbation_type="relative",
        perturbation_magnitude=5,
        scaled_range=[1.0, 2.0],
        enabled=False,
    )

    ever_config1 = EverestConfig.with_defaults(controls=[controls1])
    ever_config2 = EverestConfig.with_defaults(controls=[controls2])

    ropt_config1, initial1 = everest2ropt(
        ever_config1.controls,
        ever_config1.objective_functions,
        ever_config1.input_constraints,
        ever_config1.output_constraints,
        ever_config1.optimization,
        ever_config1.model,
        1234,
        "dummy",
    )

    ropt_config2, initial2 = everest2ropt(
        ever_config2.controls,
        ever_config2.objective_functions,
        ever_config2.input_constraints,
        ever_config2.output_constraints,
        ever_config2.optimization,
        ever_config2.model,
        1234,
        "dummy",
    )

    assert initial1 == initial2
    assert ropt_config1["names"]["variable"] == ropt_config2["names"]["variable"]
    for key in [
        "lower_bounds",
        "upper_bounds",
        "perturbation_magnitudes",
        "perturbation_types",
        "mask",
    ]:
        assert ropt_config1["variables"][key] == ropt_config2["variables"][key]
