import os
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
from pandas import Timestamp
from seba_sqlite.database import ControlDefinition, Function
from seba_sqlite.snapshot import Metadata, OptimizationInfo, SimulationInfo, Snapshot

from everest.api import EverestDataAPI
from everest.config import EverestConfig
from everest.detached import ServerStatus
from tests.everest.utils import relpath

# Global values used to create the mock snapshot.
_functions = ["f0", "f1"]
_output_constraints = [("c0", "lower_bound", 1.0), ("c1", None, None)]
_batch_ids = [0, 2]
_merit_flags = [True, False]
_obj_values = [24, 42]
_controls = [("x0", 0.1, 1.0), ("x1", None, None)]
_realizations = [1, 2, 3, 4]
_simulations = [10, 20, 30, 40, 11, 22, 33, 44]
_is_gradient = 4 * [False] + 4 * [True] + 4 * [False] + 4 * [True]
_control_values = [{"x0": 10 * sim, "x1": 20 * sim} for sim in 2 * _simulations]
_function_values = [{"f0": 100 * sim, "f1": 200 * sim} for sim in 2 * _simulations]
_dates = [Timestamp("2020-01-01"), Timestamp("2020-02-01"), Timestamp("2020-03-01")]
_weights = {"f0": 0.25, "f1": 0.75}
_norms = {"f0": 0.2, "f1": 1.5}
_gradient_info = [
    {"f0": {"x0": 10 * sim, "x1": 20 * sim}, "f1": {"x0": 30 * sim, "x1": 40 * sim}}
    for sim in 2 * _simulations
]

CONFIG_MULTIOBJ_CACHE_PATH = relpath("test_data", "cached_results_config_multiobj")
CONFIG_PATH = relpath(
    "..", "..", "test-data", "everest", "math_func", "config_multiobj.yml"
)
expected_objective_values = [-0.75, -4.75, -0.765651, -4.70363, -0.507788, -4.47678]

expected_control_values = [
    0.0,
    0.0,
    0.0,
    -0.0042039049387002465,
    -0.011301358892103368,
    1.0,
    -0.00210164760474425,
    -0.0056498598784414,
    0.499927480613782,
]

expected_single_objective_values = [-2.333333333333333, -2.333527666666667, -2.000048]


def _make_mock(mock_SebaSnapshot):
    # Build the arguments for the Snapshot constructor.
    objectives = {
        fid: Function(
            {},
            function_id=fid,
            name=fname,
            weight=_weights[fname],
            normalization=_norms[fname],
        )
        for fid, fname in enumerate(_functions)
    }
    output_constraints = {
        fid: Function(
            {},
            function_id=fid,
            name=fnc[0],
            function_type="CONSTRAINT",
            constraint_type=fnc[1],
            rhs_value=fnc[2],
        )
        for fid, fnc in enumerate(_output_constraints)
    }
    controls = {
        cid: ControlDefinition(
            {}, control_id=cid, name=name, min_value=min, max_value=max
        )
        for cid, (name, min, max) in enumerate(_controls)
    }
    functions = {
        fid: Function(
            {},
            name=name,
            weight=_weights[name],
            normalization=_norms[name],
            function_type=Function.FUNCTION_OBJECTIVE_TYPE,
        )
        for fid, name in enumerate(_functions)
    }
    metadata = Metadata(
        realizations={},
        functions=functions,
        objectives=objectives,
        constraints=output_constraints,
        controls=controls,
    )
    optimization_data = [
        OptimizationInfo(
            batch_id=bid,
            controls=None,
            objective_value=obj_val,
            merit_flag=mf,
            gradient_info=grad_val,
        )
        for bid, mf, obj_val, grad_val in zip(
            _batch_ids, _merit_flags, _obj_values, _gradient_info, strict=False
        )
    ]
    simulation_data = [
        SimulationInfo(
            batch=bid,
            objectives=fv,
            constraints=None,
            controls=con,
            sim_avg_obj=None,
            is_gradient=ig,
            realization=rea,
            start_time=None,
            end_time=None,
            success=None,
            realization_weight=1 / len(_realizations),
            simulation=sim,
        )
        for rea, sim, ig, bid, con, fv in zip(
            4 * _realizations,
            2 * _simulations,
            _is_gradient,
            8 * [_batch_ids[0]] + 8 * [_batch_ids[1]],
            _control_values,
            _function_values,
            strict=False,
        )
    ]

    # Create the reference snapshot.
    snapshot = Snapshot(
        metadata,
        [sim for sim in simulation_data if not sim.is_gradient],
        optimization_data,
    )
    snapshot_gradient = Snapshot(metadata, simulation_data, optimization_data)

    def _get_snapshot(filter_out_gradient):
        if filter_out_gradient:
            return snapshot
        else:
            return snapshot_gradient

    # Make sure the mocked get_snapshot method returns the reference snapshot.
    mock_SebaSnapshot.return_value.get_snapshot.side_effect = _get_snapshot


def _mock_summary_collector(batch):
    """
    Creates a dummy dataframe where one row contains NaN, and one column
    contains NaN. Should be filtered out before the API returns them.
    """
    simulations = [r for r in _simulations for d in _dates]
    dates = [d for r in _simulations for d in _dates]
    key1 = [batch * v for v in range(len(dates))] + [None]
    key2 = [10 * batch * v for v in range(len(dates))] + [None]
    missing_data = 25 * [None]
    dates += [Timestamp("1969-01-01")]
    simulations += [1]
    df = pd.DataFrame.from_dict(
        {
            "Realization": simulations,
            "Date": dates,
            "key1": key1,
            "key2": key2,
            "missing_data": missing_data,
        }
    )
    return df.set_index(["Realization", "Date"])


def mock_facade():
    facade = MagicMock()
    facade.enspath = "mock_path"
    return facade


def mock_storage(batches):
    storage = MagicMock()
    data = MagicMock()
    data.load_all_summary_data.side_effect = [
        _mock_summary_collector(i) for i in batches
    ]
    experiment = storage.get_experiment_by_name.return_value
    experiment.get_ensemble_by_name.return_value = data
    return storage


@pytest.fixture
@patch.object(EverestConfig, "optimization_output_dir", new_callable=PropertyMock)
@patch("everest.api.everest_data_api.SebaSnapshot")
def api(mock_SebaSnapshot, mock_get_optimization_output_dir):
    _make_mock(mock_SebaSnapshot)
    return EverestDataAPI(EverestConfig.with_defaults(), filter_out_gradient=False)


@pytest.fixture
@patch.object(EverestConfig, "optimization_output_dir", new_callable=PropertyMock)
@patch("everest.api.everest_data_api.SebaSnapshot")
def api_no_gradient(mock_SebaSnapshot, mock_get_optimization_output_dir):
    _make_mock(mock_SebaSnapshot)
    return EverestDataAPI(EverestConfig.with_defaults())


def test_batches(api_no_gradient, api):
    assert api_no_gradient.batches == _batch_ids
    assert api.batches == _batch_ids


def test_accepted_batches(api_no_gradient, api):
    expected_result = [
        bid for bid, mf in zip(_batch_ids, _merit_flags, strict=False) if mf
    ]
    assert api_no_gradient.accepted_batches == expected_result
    assert api.accepted_batches == expected_result


def test_function_names(api_no_gradient, api):
    assert api_no_gradient.objective_function_names == _functions
    assert api.objective_function_names == _functions


def test_output_constraint_names(api_no_gradient, api):
    expected_result = [oc[0] for oc in _output_constraints]
    assert api_no_gradient.output_constraint_names == expected_result
    assert api.output_constraint_names == expected_result


def test_input_constraints(api_no_gradient, api):
    for control in _controls:
        expected_result = {"min": control[1], "max": control[2]}
        result = api_no_gradient.input_constraint(control[0])
        assert result == expected_result
        assert api.input_constraint(control[0]) == expected_result


def test_output_constraints(api_no_gradient, api):
    for constraint in _output_constraints:
        expected_result = {"type": constraint[1], "right_hand_side": constraint[2]}
        result = api_no_gradient.output_constraint(constraint[0])
        assert result == expected_result
        assert api.output_constraint(constraint[0]) == expected_result


def test_realizations(api_no_gradient, api):
    assert api_no_gradient.realizations == _realizations
    assert api.realizations == _realizations


def test_simulations(api_no_gradient, api):
    expected_result = [
        s for s, g in zip(_simulations, _is_gradient, strict=False) if not g
    ]
    assert api_no_gradient.simulations == expected_result
    assert api.simulations == _simulations


def test_control_names(api_no_gradient, api):
    expected_result = [con[0] for con in _controls]
    assert api_no_gradient.control_names == expected_result
    assert api.control_names == expected_result


def test_control_values(api_no_gradient, api):
    control_names = api_no_gradient.control_names
    expected_result = [
        {"batch": bid, "control": name, "value": con[name]}
        for bid, con in zip(
            8 * [_batch_ids[0]] + 8 * [_batch_ids[1]], _control_values, strict=False
        )
        for name in control_names
    ]
    for res, er in zip(api.control_values, expected_result, strict=False):
        assert res == er
    is_gradient = [ig for ig in _is_gradient for name in control_names]
    expected_result = [
        tv for tv, ig in zip(expected_result, is_gradient, strict=False) if not ig
    ]
    for res, er in zip(api_no_gradient.control_values, expected_result, strict=False):
        assert res == er


def test_objective_values(api_no_gradient, api):
    function_names = api_no_gradient.objective_function_names
    expected_result = [
        {
            "batch": bid,
            "realization": rea,
            "function": name,
            "value": fnc[name],
            "simulation": sim,
            "weight": _weights[name],
            "norm": _norms[name],
        }
        for rea, sim, bid, fnc in zip(
            4 * _realizations,
            2 * _simulations,
            8 * [_batch_ids[0]] + 8 * [_batch_ids[1]],
            _function_values,
            strict=False,
        )
        for name in function_names
    ]
    for res, er in zip(api.objective_values, expected_result, strict=False):
        assert res == er
    is_gradient = [ig for ig in _is_gradient for name in function_names]
    expected_result = [
        tv for tv, ig in zip(expected_result, is_gradient, strict=False) if not ig
    ]
    for res, er in zip(api_no_gradient.objective_values, expected_result, strict=False):
        assert res == er


def test_single_objective_values(api_no_gradient):
    function_data = [
        (rea, bid, fv)
        for rea, ig, bid, fv in zip(
            4 * _realizations,
            _is_gradient,
            8 * [_batch_ids[0]] + 8 * [_batch_ids[1]],
            _function_values,
            strict=False,
        )
        if not ig
    ]
    expected_objectives = {b: {} for b in _batch_ids}
    realization_weight = 1 / len(_realizations)
    for _r, b, f in function_data:
        for name in _functions:
            factor = realization_weight * _weights[name] * _norms[name]
            if expected_objectives[b].get(name, None) is None:
                expected_objectives[b].update({name: f[name] * factor})
            else:
                expected_objectives[b][name] += f[name] * factor

    expected_result = [
        {"accepted": m, "batch": b, "objective": v, **expected_objectives[b]}
        for b, v, m in zip(_batch_ids, _obj_values, _merit_flags, strict=False)
    ]

    result = api_no_gradient.single_objective_values
    assert result == expected_result


def test_gradient_values(api_no_gradient):
    expected_result = [
        {"batch": bid, "function": func, "control": ctrl, "value": val}
        for bid, grad_info in zip(_batch_ids, _gradient_info, strict=False)
        for func, info in grad_info.items()
        for ctrl, val in info.items()
    ]
    assert api_no_gradient.gradient_values == expected_result


@patch.object(
    EverestConfig,
    "optimization_output_dir",
    new_callable=PropertyMock,
    return_value=CONFIG_MULTIOBJ_CACHE_PATH,
)
def test_ropt_integration(mock_get_optimization_output_dir):
    config = EverestConfig.load_file(CONFIG_PATH)
    api = EverestDataAPI(config)

    expected_objectives = [
        {"name": f.name, "weight": f.weight, "norm": f.normalization}
        for f in config.objective_functions
    ]
    expected_objective_names = [objective["name"] for objective in expected_objectives]
    assert expected_objective_names == api.objective_function_names

    expected_weight = [objective["weight"] for objective in expected_objectives]
    weight_sum = sum(expected_weight)
    if weight_sum != 1:
        for obj in expected_objectives:
            obj["weight"] /= weight_sum

    for obj in api.objective_values:
        res = {
            "name": obj["function"],
            "weight": obj["weight"],
            "norm": obj["norm"],
        }
        assert res in expected_objectives

    batches = [0, 1, 2]
    assert batches == api.batches

    accepted_batches = [0, 2]
    assert accepted_batches == api.accepted_batches

    assert api.output_constraint_names == []

    realizations = config.model.realizations
    assert realizations == api.realizations

    expected_control_names = []
    for control in config.controls:
        expected_control_names += [
            f"{control.name}_{var.name}" for var in control.variables
        ]
    assert expected_control_names == api.control_names

    expected_control_constraint = {"min": -1.0, "max": 1.0}
    for name in expected_control_names:
        assert expected_control_constraint == api.input_constraint(name)

    for name in {c["control"] for c in api.control_values}:
        assert name in expected_control_names

    for value in [c["value"] for c in api.control_values]:
        assert value in expected_control_values

    expected_batches = [0, 1, 2]
    assert expected_batches == list({c["batch"] for c in api.control_values})

    for name in {f["function"] for f in api.objective_values}:
        assert name in expected_objective_names

    result = list({f["batch"] for f in api.objective_values})

    assert expected_batches == result

    for value in [f["value"] for f in api.objective_values]:
        assert value in expected_objective_values

    for value in [f["objective"] for f in api.single_objective_values]:
        assert value in expected_single_objective_values

    assert batches == [obj["batch"] for obj in api.single_objective_values]


@patch("everest.api.everest_data_api.open_storage", return_value=mock_storage((0, 2)))
def test_get_summary_keys(_, api_no_gradient):
    # Retrieve the pandas data frame with mocked values.
    summary_values = api_no_gradient.summary_values()
    # Check some data frame properties.
    assert set(summary_values.columns) == {
        "realization",
        "simulation",
        "batch",
        "date",
        "key1",
        "key2",
    }
    assert summary_values.shape[0] == len(_realizations) * len(_batch_ids) * len(_dates)
    assert set(summary_values["batch"]) == set(_batch_ids)
    assert set(summary_values["realization"]) == set(_realizations)
    non_gradient_simulations = [
        s for s, g in zip(_simulations, _is_gradient, strict=False) if not g
    ]
    assert set(summary_values["simulation"]) == set(non_gradient_simulations)
    assert set(summary_values["date"]) == set(_dates)
    # Check key values.
    for batch_id in _batch_ids:
        batch = summary_values.loc[summary_values["batch"] == batch_id]
        assert batch["key1"].to_list() == [batch_id * v for v in range(batch.shape[0])]


@patch("everest.api.everest_data_api.open_storage", return_value=mock_storage((0, 2)))
def test_get_summary_keys_gradient(_, api):
    # Retrieve the pandas data frame with mocked values.
    summary_values = api.summary_values()
    # Check some data frame properties.
    assert set(summary_values.columns) == {
        "realization",
        "simulation",
        "batch",
        "date",
        "key1",
        "key2",
    }
    assert summary_values.shape[0] == len(_simulations) * len(_batch_ids) * len(_dates)
    assert set(summary_values["batch"]) == set(_batch_ids)
    assert set(summary_values["realization"]) == set(_realizations)
    assert set(summary_values["simulation"]) == set(_simulations)
    assert set(summary_values["date"]) == set(_dates)
    # Check key values.
    for batch_id in _batch_ids:
        batch = summary_values.loc[summary_values["batch"] == batch_id]
        assert batch["key1"].to_list() == [batch_id * v for v in range(batch.shape[0])]


@patch("everest.api.everest_data_api.open_storage", return_value=mock_storage([2]))
def test_get_summary_keys_single_batch(_, api_no_gradient):
    # Retrieve the pandas data frame with mocked values.
    summary_values = api_no_gradient.summary_values(batches=[2])
    # Check some data frame properties.
    assert set(summary_values.columns) == {
        "realization",
        "simulation",
        "batch",
        "date",
        "key1",
        "key2",
    }
    assert summary_values.shape[0] == len(_realizations) * len(_dates)
    assert summary_values["batch"].iloc[0] == 2
    assert set(summary_values["realization"]) == set(_realizations)
    non_gradient_simulations = [
        s for s, g in zip(_simulations, _is_gradient, strict=False) if not g
    ]
    assert set(summary_values["simulation"]) == set(non_gradient_simulations)
    assert set(summary_values["date"]) == set(_dates)

    # Check key values.
    batch = summary_values.loc[summary_values["batch"] == 2]
    assert batch["key1"].to_list() == [2 * v for v in range(batch.shape[0])]
    assert batch["key2"].to_list() == [20 * v for v in range(batch.shape[0])]


@patch("everest.api.everest_data_api.open_storage", return_value=mock_storage((0, 2)))
def test_get_summary_keys_single_key(_, api_no_gradient):
    # Retrieve the pandas data frame with mocked values.
    summary_values = api_no_gradient.summary_values(keys=["key2"])
    # Check some data frame properties.
    assert set(summary_values.columns) == {
        "realization",
        "simulation",
        "batch",
        "date",
        "key2",
    }
    assert summary_values.shape[0] == len(_realizations) * len(_batch_ids) * len(_dates)
    assert set(summary_values["batch"]) == set(_batch_ids)
    assert set(summary_values["realization"]) == set(_realizations)
    non_gradient_simulations = [
        s for s, g in zip(_simulations, _is_gradient, strict=False) if not g
    ]
    assert set(summary_values["simulation"]) == set(non_gradient_simulations)
    assert set(summary_values["date"]) == set(_dates)
    # Check key values.
    for batch_id in _batch_ids:
        batch = summary_values.loc[summary_values["batch"] == batch_id]
        assert batch["key2"].to_list() == [
            10 * batch_id * v for v in range(batch.shape[0])
        ]


@patch.object(EverestConfig, "optimization_output_dir", new_callable=PropertyMock)
@patch("everest.api.everest_data_api.SebaSnapshot")
@patch("everest.api.everest_data_api.SebaSnapshot.get_snapshot")
def test_output_folder(_1, _2, _3, copy_math_func_test_data_to_tmp):
    config_file = "config_multiobj.yml"
    config = EverestConfig.load_file(config_file)
    assert config.environment is not None
    expected = config.environment.output_folder
    api = EverestDataAPI(config)
    assert expected == os.path.basename(api.output_folder)


@patch.object(EverestConfig, "optimization_output_dir", new_callable=PropertyMock)
@patch("everest.api.everest_data_api.SebaSnapshot")
@patch("everest.api.everest_data_api.SebaSnapshot.get_snapshot")
@patch(
    "everest.api.everest_data_api.everserver_status",
    return_value={"status": ServerStatus.completed},
)
def test_everest_csv(
    everserver_status_mock, _1, _2, _3, copy_math_func_test_data_to_tmp
):
    config_file = "config_multiobj.yml"
    config = EverestConfig.load_file(config_file)
    expected = config.export_path
    api = EverestDataAPI(config)
    assert expected == api.everest_csv

    everserver_status_mock.return_value = {"status": ServerStatus.running}
    assert api.everest_csv is None
