import numpy as np
import polars as pl
import pytest

from ert.config import GenKwConfig
from ert.storage import open_storage


def test_that_load_scalar_keys_loads_all_parameters(tmp_path):
    """Test that load_scalar_keys loads all scalar parameters when keys=None."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        # Save parameters
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load all parameters
        df = ensemble.load_scalar_keys()
        assert df.shape == (3, 4)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param2" in df.columns
        assert "param3" in df.columns
        assert df["param1"].to_list() == [1.0, 2.0, 3.0]


def test_that_load_scalar_keys_loads_specific_parameters(tmp_path):
    """Test that load_scalar_keys loads only specified parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load only param1 and param3
        df = ensemble.load_scalar_keys(keys=["param1", "param3"])
        assert df.shape == (3, 3)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param3" in df.columns
        assert "param2" not in df.columns


def test_that_load_scalar_keys_filters_by_realizations(tmp_path):
    """Test that load_scalar_keys filters by specified realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2, 3, 4],
                    "param1": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        )

        # Load only realizations 1 and 3
        df = ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([1, 3]))
        assert df.shape == (2, 2)
        assert df["realization"].to_list() == [1, 3]
        assert df["param1"].to_list() == [2.0, 4.0]


def test_that_load_scalar_keys_raises_key_error_for_missing_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for non-existent parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        with pytest.raises(KeyError, match="No SCALAR dataset in storage"):
            ensemble.load_scalar_keys(keys=["param1"])


def test_that_load_scalar_keys_raises_key_error_for_unregistered_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for parameters not in experiment."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            KeyError,
            match="Parameters not registered to the experiment: \\{'param2'\\}",
        ):
            ensemble.load_scalar_keys(keys=["param1", "param2"])


def test_that_load_scalar_keys_raises_index_error_for_missing_realizations(tmp_path):
    """Test that load_scalar_keys raises IndexError when no matching realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            IndexError,
            match="No matching realizations \\[5 6\\] found for \\['param1'\\]",
        ):
            ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([5, 6]))
