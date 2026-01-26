from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from ert.config import EverestControl


def test_that_write_to_runpath_writes_json_with_correct_structure(tmp_path):
    control = EverestControl(
        name="point",
        input_keys=["point.x", "point.y", "point.z"],
        output_file="point.json",
        types=["generic_control", "generic_control", "generic_control"],
        initial_guesses=[0.1, 0.1, 0.1],
        control_types=["real", "real", "real"],
        enabled=[True, True, True],
        min=[-1.0, -1.0, -1.0],
        max=[1.0, 1.0, 1.0],
        perturbation_types=["absolute", "absolute", "absolute"],
        perturbation_magnitudes=[0.1, 0.1, 0.1],
        scaled_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        samplers=[None, None, None],
    )

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_df = pl.DataFrame(
        {
            "realization": [5],
            "point.x": [1.5],
            "point.y": [2.5],
            "point.z": [3.5],
        }
    )
    mock_ensemble.load_parameters.return_value = mock_df

    run_path = tmp_path / "runpath" / "realization-5"

    control.write_to_runpath(run_path, 5, mock_ensemble)

    mock_ensemble.load_parameters.assert_called_once_with("point", 5)

    output_file = run_path / "point.json"
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert data == {"x": 1.5, "y": 2.5, "z": 3.5}


def test_that_write_to_runpath_writes_json_with_correct_structure_for_nested_controls(
    tmp_path,
):
    control = EverestControl(
        name="point",
        input_keys=["point.0.x", "point.1.x", "point.2.x"],
        output_file="point.json",
        types=["generic_control", "generic_control", "generic_control"],
        initial_guesses=[0.1, 0.1, 0.1],
        control_types=["real", "real", "real"],
        enabled=[True, True, True],
        min=[-1.0, -1.0, -1.0],
        max=[1.0, 1.0, 1.0],
        perturbation_types=["absolute", "absolute", "absolute"],
        perturbation_magnitudes=[0.1, 0.1, 0.1],
        scaled_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        samplers=[None, None, None],
    )

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0
    mock_df = pl.DataFrame(
        {
            "realization": [5],
            "point.x.0": [1.5],
            "point.x.1": [2.5],
            "point.x.2": [3.5],
        }
    )
    mock_ensemble.load_parameters.return_value = mock_df

    run_path = tmp_path / "runpath" / "realization-5"
    control.write_to_runpath(run_path, 5, mock_ensemble)

    mock_ensemble.load_parameters.assert_called_once_with("point", 5)
    output_file = run_path / "point.json"
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert data == {"x": {"0": 1.5, "1": 2.5, "2": 3.5}}


def test_that_create_storage_datasets_returns_dataframe_with_correct_schema():
    control = EverestControl(
        name="point",
        input_keys=["point.x", "point.y", "point.z"],
        output_file="point.json",
        types=["generic_control", "generic_control", "generic_control"],
        initial_guesses=[0.1, 0.1, 0.1],
        control_types=["real", "real", "real"],
        enabled=[True, True, True],
        min=[-1.0, -1.0, -1.0],
        max=[1.0, 1.0, 1.0],
        perturbation_types=["absolute", "absolute", "absolute"],
        perturbation_magnitudes=[0.1, 0.1, 0.1],
        scaled_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        samplers=[None, None, None],
    )

    from_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iens_active_index = np.array([0, 1])

    result = list(control.create_storage_datasets(from_data, iens_active_index))

    assert len(result) == 1
    i, df = result[0]
    assert i is None
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["realization", "point.x", "point.y", "point.z"]
    assert df.shape == (2, 4)


def test_that_create_storage_datasets_preserves_data_values():
    control = EverestControl(
        name="point",
        input_keys=["point.x", "point.y", "point.z"],
        output_file="point.json",
        types=["generic_control", "generic_control", "generic_control"],
        initial_guesses=[0.1, 0.1, 0.1],
        control_types=["real", "real", "real"],
        enabled=[True, True, True],
        min=[-1.0, -1.0, -1.0],
        max=[1.0, 1.0, 1.0],
        perturbation_types=["absolute", "absolute", "absolute"],
        perturbation_magnitudes=[0.1, 0.1, 0.1],
        scaled_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        samplers=[None, None, None],
    )

    from_data = np.array([[10.5, 20.3, 30.1], [30.7, 40.9, 50.2], [50.1, 60.2, 70.3]])
    iens_active_index = np.array([5, 10, 15])

    result = list(control.create_storage_datasets(from_data, iens_active_index))
    i, df = result[0]
    assert i is None
    assert df["realization"].to_list() == [5, 10, 15]
    assert df["point.x"].to_list() == [10.5, 30.7, 50.1]
    assert df["point.y"].to_list() == [20.3, 40.9, 60.2]
    assert df["point.z"].to_list() == [30.1, 50.2, 70.3]


def test_that_create_storage_datasets_handles_nested_parameter_keys():
    control = EverestControl(
        name="point",
        input_keys=["point.0.x", "point.1.x", "point.2.x"],
        output_file="point.json",
        types=["generic_control", "generic_control", "generic_control"],
        initial_guesses=[0.1, 0.1, 0.1],
        control_types=["real", "real", "real"],
        enabled=[True, True, True],
        min=[-1.0, -1.0, -1.0],
        max=[1.0, 1.0, 1.0],
        perturbation_types=["absolute", "absolute", "absolute"],
        perturbation_magnitudes=[0.1, 0.1, 0.1],
        scaled_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        samplers=[None, None, None],
    )

    from_data = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    iens_active_index = np.array([0, 1])

    result = list(control.create_storage_datasets(from_data, iens_active_index))
    _, df = result[0]

    assert df.columns == ["realization", "point.0.x", "point.1.x", "point.2.x"]
    assert df["point.0.x"].to_list() == [1.0, 5.0]
    assert df["point.1.x"].to_list() == [2.0, 6.0]
    assert df["point.2.x"].to_list() == [3.0, 7.0]
