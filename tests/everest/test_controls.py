import itertools
import numbers
import os
from copy import deepcopy
from typing import List

import pytest
from pydantic import ValidationError

from everest import ConfigKeys
from everest.config import EverestConfig
from everest.config.control_config import ControlConfig
from everest.config.control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from everest.config.input_constraint_config import InputConstraintConfig
from everest.config.well_config import WellConfig
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
            type=ConfigKeys.WELL_CONTROL,
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
        EverestConfig.model_validate(config.model_dump(exclude_none=True))

    config.controls[1].name = exp_grp_name + "_new"
    EverestConfig.model_validate(config.model_dump(exclude_none=True))


def _perturb_control_zero(
    config: EverestConfig, gmin, gmax, ginit, fill
) -> List[ControlVariableConfig]:
    """Perturbs the variable range of the first control to create
    interesting configurations.
    """
    control_zero = config.controls[0]
    variable_names = [var.name for var in control_zero.variables]
    revised_control_zero = None

    exp_var_def = []
    for idx, var_name in enumerate(variable_names):
        var_config = ControlVariableConfig.model_validate({"name": var_name})

        if idx % 2 == 0 or fill:
            var_config.min = gmin - 0.3 if gmin else 0.13
        if idx % 3 == 0 or fill:
            var_config.max = gmax + 1.2 if gmax else 2.64
        if idx % 4 == 0 or fill:
            var_max = var_config.max if var_config.max is not None else gmax
            var_min = var_config.min if var_config.min is not None else gmin
            if var_min and var_max:
                var_config.initial_guess = (var_min + var_max) / 2.0
        if revised_control_zero is None:
            revised_control_zero = ControlConfig(
                name=control_zero.name,
                type=control_zero.type,
                min=gmin,
                max=gmax,
                initial_guess=ginit,
                variables=[var_config],
            )
        else:
            revised_control_zero.variables.append(var_config)

        exp_var_def.append(
            ControlVariableConfig(
                name=var_name,
                min=var_config.min if var_config.min is not None else gmin,
                max=var_config.max if var_config.max is not None else gmax,
                initial_guess=(
                    var_config.initial_guess
                    if var_config.initial_guess is not None
                    else ginit
                ),
            )
        )

    config.controls[0] = revised_control_zero

    return exp_var_def


def test_variable_name_index_validation(copy_test_data_to_tmp):
    config = EverestConfig.load_file(
        os.path.join("mocked_test_case", "mocked_test_case.yml")
    )

    # Not valid equal names
    config.controls[0].variables[1].name = "w00"
    with pytest.raises(
        ValidationError, match=r"Subfield\(s\) `name-index` must be unique"
    ):
        EverestConfig.model_validate(config.model_dump(exclude_none=True))
    # Not valid index inconsistency
    config.controls[0].variables[1].name = "w01"
    config.controls[0].variables[1].index = 0
    with pytest.raises(
        ValidationError, match="for all of the variables or for none of them"
    ):
        EverestConfig.model_validate(config.model_dump(exclude_none=True))

    # Index and name not unique
    config.controls[0].variables[1].name = "w00"
    for v in config.controls[0].variables:
        v.index = 0
    with pytest.raises(
        ValidationError, match=r"Subfield\(s\) `name-index` must be unique"
    ):
        EverestConfig.model_validate(config.model_dump(exclude_none=True))

    # Index and name unique and valid, but input constraints are not
    # specifying index

    config.controls[0].variables[1].name = "w01"
    input_constraints = [
        InputConstraintConfig.model_validate(
            {"upper_bound": 1, "lower_bound": 0, "weights": {"group.w00": 0.1}}
        )
    ]

    config.input_constraints = input_constraints
    with pytest.raises(
        ValidationError,
        match="does not match any instance of "
        "control_name.variable_name-variable_index",
    ):
        EverestConfig.model_validate(config.model_dump(exclude_none=True))

    # Index and name unique and valid and input constraints are specifying
    # index
    input_constraints = [
        InputConstraintConfig(
            **{"upper_bound": 1, "lower_bound": 0, "weights": {"group.w00-0": 0.1}}
        )
    ]

    config.input_constraints = input_constraints
    EverestConfig.model_validate(config.model_dump(exclude_none=True))


@pytest.mark.integration_test
def test_individual_control_variable_config(copy_test_data_to_tmp):
    config_file = os.path.join("mocked_test_case", "config_input_constraints.yml")

    global_min = (0, 0.7, 1.3, None)
    global_max = (0.5, 1, 3, None)
    global_init = (0.3, 0.6, 1.1, None)
    fill_missing = (True, False)
    test_base = (global_min, global_max, global_init, fill_missing)

    for gmin, gmax, ginit, fill in itertools.product(*test_base):
        config = EverestConfig.load_file(config_file)
        exp_var_def = _perturb_control_zero(config, gmin, gmax, ginit, fill)

        # Not complete configuration
        if None in [gmin, gmax, ginit] and not fill:
            with pytest.raises(expected_exception=ValidationError):
                EverestConfig.model_validate(config.to_dict())
            continue

        # Invalid parameters
        def valid_control(var: ControlVariableConfig) -> bool:
            return var.min <= var.initial_guess <= var.max

        if not all(map(valid_control, exp_var_def)):
            with pytest.raises(expected_exception=ValidationError):
                EverestConfig.model_validate(config.to_dict())
            continue

        EverestConfig.model_validate(config.to_dict())


def test_control_variable_name():
    """We would potentially like to support variable names with
    underscores, but currently Seba is using this as a separator between
    the group name and the variable name in such a way that having an
    underscore in a variable name will not behave nicely..
    """
    config = EverestConfig.load_file(mocked_config)
    EverestConfig.model_validate(config.model_dump(exclude_none=True))

    illegal_name = "illegal.name.due.to.dots"
    config.controls[0].variables[0].name = illegal_name
    with pytest.raises(
        ValidationError,
        match="Variable name can not contain any dots",
    ):
        EverestConfig.model_validate(config.model_dump(exclude_none=True))

    weirdo_name = "something/with-symbols_=/()*&%$#!"
    new_config = EverestConfig.load_file(mocked_config)
    new_config.wells.append(WellConfig(name=weirdo_name))
    new_config.controls[0].variables[0].name = weirdo_name
    EverestConfig.model_validate(new_config.model_dump(exclude_none=True))


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
