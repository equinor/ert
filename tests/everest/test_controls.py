import numbers
from copy import deepcopy

import pytest
from pydantic import ValidationError

from ert.config import ConfigWarning
from everest.config import EverestConfig, InputConstraintConfig
from everest.config.control_config import ControlConfig
from everest.config.control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from everest.config.well_config import WellConfig
from everest.optimizer.everest2ropt import everest2ropt
from tests.everest.utils import relpath

cfg_dir = relpath("test_data", "mocked_test_case")
mocked_config = relpath(cfg_dir, "mocked_test_case.yml")


def test_controls_initialization():
    exp_grp_name = "group"

    config = EverestConfig.load_file(mocked_config)
    assert config.controls is not None
    group = config.controls[0]
    assert group.variables is not None

    assert exp_grp_name == group.name

    for c in group.variables:
        assert isinstance(c.name, str)
        assert isinstance(c.initial_guess, numbers.Number)

    a_ctrl_name = group.variables[0].name
    config.controls.append(
        ControlConfig(
            name=exp_grp_name,
            type="well_control",
            variables=[
                ControlVariableConfig(
                    name=a_ctrl_name,
                    min=0,
                    max=1,
                    initial_guess=0.5,
                )
            ],
        )
    )
    with pytest.raises(
        ValidationError,
        match=r"Subfield\(s\) `name` must be unique",
    ):
        EverestConfig.model_validate(config.to_dict())

    config.controls[1].name = exp_grp_name + "_new"
    EverestConfig.model_validate(config.to_dict())


def test_control_variable_duplicate_name_no_index():
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


def test_control_variable_index_inconsistency():
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


def test_control_variable_duplicate_name_and_index():
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


def test_input_constraint_name_mismatch_with_indexed_variables():
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


def test_input_constraint_deprecated_indexed_name_format_warns():
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


def test_control_variable_initial_guess_below_min():
    with pytest.raises(ValidationError, match="initial_guess"):
        ControlConfig(
            name="control",
            type="well_control",
            variables=[
                ControlVariableConfig(name="w00", min=0.5, max=1.0, initial_guess=0.3)
            ],
        )


def test_control_variable_initial_guess_above_max():
    with pytest.raises(ValidationError, match="initial_guess"):
        ControlConfig(
            name="control",
            type="well_control",
            variables=[
                ControlVariableConfig(name="w00", min=0.5, max=1.0, initial_guess=1.3)
            ],
        )


def test_control_variable_name():
    """We would potentially like to support variable names with
    underscores, but currently Seba is using this as a separator between
    the group name and the variable name in such a way that having an
    underscore in a variable name will not behave nicely..
    """
    config = EverestConfig.load_file(mocked_config)
    EverestConfig.model_validate(config.to_dict())

    illegal_name = "illegal.name.due.to.dots"
    config.controls[0].variables[0].name = illegal_name
    with pytest.raises(
        ValidationError,
        match="Variable name can not contain any dots",
    ):
        EverestConfig.model_validate(config.to_dict())

    weirdo_name = "something/with-symbols_=/()*&%$#!"
    new_config = EverestConfig.load_file(mocked_config)
    new_config.wells.append(WellConfig(name=weirdo_name))
    new_config.controls[0].variables[0].name = weirdo_name
    EverestConfig.model_validate(new_config.to_dict())


def test_control_none_well_variable_name():
    config = EverestConfig.load_file(mocked_config)
    EverestConfig.model_validate(config.to_dict())

    illegal_name = "nowell4sure"
    config.controls[0].variables[0].name = illegal_name
    with pytest.raises(
        ValidationError,
        match="Variable name does not match any well name",
    ):
        EverestConfig.model_validate(config.to_dict())


def test_control_variable_types(control_config: ControlConfig):
    if isinstance(control_config.variables[0], ControlVariableConfig):
        assert all(
            isinstance(variable, ControlVariableConfig)
            for variable in control_config.variables
        )
    else:
        assert all(
            isinstance(variable, ControlVariableGuessListConfig)
            for variable in control_config.variables
        )


@pytest.mark.parametrize(
    "variables",
    (
        pytest.param(
            [
                {"name": "w00", "initial_guess": 0.0626, "index": 0},
                {"name": "w00", "initial_guess": [0.063, 0.0617, 0.0621]},
            ],
            id="same name",
        ),
        pytest.param(
            [
                {"name": "w00", "initial_guess": 0.0626, "index": 0},
                {"name": "w01", "initial_guess": [0.0627, 0.0631, 0.0618, 0.0622]},
            ],
            id="different name",
        ),
    ),
)
def test_control_bad_variables(variables, control_data_no_variables: dict):
    data = deepcopy(control_data_no_variables)
    data["variables"] = variables
    with pytest.raises(ValidationError, match="3 validation errors"):
        ControlConfig.model_validate(data)


def test_control_variable_guess_list():
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
