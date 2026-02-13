from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from ert.config import EverestControl
from ert.run_models._create_run_path import _generate_parameter_files


def test_that_write_to_runpath_writes_json_with_correct_structure(tmp_path):
    controls = [
        EverestControl(
            name=f"point.{coord}",
            input_key=f"point.{coord}",
            output_file="point.json",
            group="point",
            control_type_="generic_control",
            initial_guess=0.1,
            control_type="real",
            enabled=True,
            min=-1.0,
            max=1.0,
            perturbation_type="absolute",
            perturbation_magnitude=0.1,
            scaled_range=(-1.0, 1.0),
            sampler=None,
        )
        for coord in ["x", "y", "z"]
    ]

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0

    mock_ensemble.load_parameters.side_effect = [
        pl.DataFrame({"realization": [5], "point.x": [1.5]}),
        pl.DataFrame({"realization": [5], "point.y": [2.5]}),
        pl.DataFrame({"realization": [5], "point.z": [3.5]}),
    ]

    run_path = tmp_path / "runpath" / "realization-5"

    run_path.mkdir(parents=True)

    _generate_parameter_files(
        parameter_configs=controls,
        export_base_name="parameters",
        run_path=run_path,
        iens=5,
        fs=mock_ensemble,
        iteration=0,
    )

    output_file = run_path / "point.json"
    output_file.write_text(
        json.dumps(json.loads(output_file.read_text()), sort_keys=True),
        encoding="utf-8",
    )  # Normalize for comparison if needed, or rely on dict equality

    assert output_file.exists()
    result_data = json.loads(output_file.read_text())
    assert result_data == {"x": 1.5, "y": 2.5, "z": 3.5}


def test_that_write_to_runpath_writes_json_with_correct_structure_for_nested_controls(
    tmp_path,
):
    controls = [
        EverestControl(
            name=f"point.x.{i}",
            input_key=f"point.x.{i}",
            output_file="point.json",
            group="point",
            control_type_="generic_control",
            initial_guess=0.1,
            control_type="real",
            enabled=True,
            min=-1.0,
            max=1.0,
            perturbation_type="absolute",
            perturbation_magnitude=0.1,
            scaled_range=(-1.0, 1.0),
            sampler=None,
        )
        for i in range(3)
    ]

    mock_ensemble = MagicMock()
    mock_ensemble.iteration = 0

    mock_ensemble.load_parameters.side_effect = [
        pl.DataFrame({"realization": [5], "point.x.0": [1.5]}),
        pl.DataFrame({"realization": [5], "point.x.1": [2.5]}),
        pl.DataFrame({"realization": [5], "point.x.2": [3.5]}),
    ]

    run_path = tmp_path / "runpath" / "realization-5"
    run_path.mkdir(parents=True)

    _generate_parameter_files(
        parameter_configs=controls,
        export_base_name="parameters",
        run_path=run_path,
        iens=5,
        fs=mock_ensemble,
        iteration=0,
    )

    output_file = run_path / "point.json"
    assert output_file.exists()
    result_data = json.loads(output_file.read_text())
    assert result_data == {"x": {"0": 1.5, "1": 2.5, "2": 3.5}}


def test_that_create_storage_datasets_returns_dataframe_with_correct_schema():
    control = EverestControl(
        name="point.x",
        input_key="point.x",
        output_file="point.json",
        control_type_="generic_control",
        initial_guess=0.1,
        control_type="real",
        enabled=True,
        min=-1.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.1,
        scaled_range=(-1.0, 1.0),
        sampler=None,
        group="point",
    )

    from_data = np.array([[1.0], [4.0]])
    iens_active_index = np.array([0, 1])

    result = list(control.create_storage_datasets(from_data, iens_active_index))

    assert len(result) == 1
    i, df = result[0]
    assert i is None
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["realization", "point.x"]
    assert df.shape == (2, 2)


def test_that_create_storage_datasets_preserves_data_values():
    control = EverestControl(
        name="point.x",
        input_key="point.x",
        output_file="point.json",
        control_type_="generic_control",
        initial_guess=0.1,
        control_type="real",
        enabled=True,
        min=-1.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.1,
        scaled_range=(-1.0, 1.0),
        sampler=None,
        group="point",
    )

    from_data = np.array([[10.5], [30.7], [50.1]])
    iens_active_index = np.array([5, 10, 15])

    result = list(control.create_storage_datasets(from_data, iens_active_index))
    i, df = result[0]
    assert i is None
    assert df["realization"].to_list() == [5, 10, 15]
    assert df["point.x"].to_list() == [10.5, 30.7, 50.1]


def test_that_create_storage_datasets_handles_nested_parameter_keys():
    control = EverestControl(
        name="point.0.x",
        input_key="point.0.x",
        output_file="point.json",
        control_type_="generic_control",
        initial_guess=0.1,
        control_type="real",
        enabled=True,
        min=-1.0,
        max=1.0,
        perturbation_type="absolute",
        perturbation_magnitude=0.1,
        scaled_range=(-1.0, 1.0),
        sampler=None,
        group="point",
    )

    from_data = np.array([[1.0], [5.0]])
    iens_active_index = np.array([0, 1])

    result = list(control.create_storage_datasets(from_data, iens_active_index))
    _, df = result[0]

    assert df.columns == ["realization", "point.0.x"]
    assert df["point.0.x"].to_list() == [1.0, 5.0]
