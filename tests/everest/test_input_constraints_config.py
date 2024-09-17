from everest import ConfigKeys
from everest.config import EverestConfig
from tests.everest.utils import relpath


def test_input_constraint_initialization():
    cfg_dir = relpath("test_data", "mocked_test_case")
    cfg = relpath(cfg_dir, "config_input_constraints.yml")
    config = EverestConfig.load_file(cfg)
    # Check that an input constraint has been defined
    assert config.input_constraints is not None
    # Check that it is a list with two values
    assert isinstance(config.input_constraints, list)
    assert len(config.input_constraints) == 2
    # Get the first input constraint
    input_constraint = config.input_constraints[0]
    # Check that this defines both upper and lower bounds
    exp_operations = {ConfigKeys.UPPER_BOUND, ConfigKeys.LOWER_BOUND}
    assert (
        exp_operations.intersection(input_constraint.model_dump(exclude_none=True))
        == exp_operations
    )
    # Check both rhs
    exp_rhs = [1, 0]
    assert [
        input_constraint.upper_bound,
        input_constraint.lower_bound,
    ] == exp_rhs
    # Check the variables
    exp_vars = ["group.w00", "group.w01", "group.w02"]
    assert set(exp_vars) == set(input_constraint.weights.keys())
    # Check the weights
    exp_weights = [0.1, 0.2, 0.3]
    assert exp_weights == [input_constraint.weights[v] for v in exp_vars]
