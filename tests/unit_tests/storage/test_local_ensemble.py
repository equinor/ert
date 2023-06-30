import numpy as np
import xarray as xr
import xtgeo
from ecl.grid import EclGridGenerator

from ert.storage import open_storage
from ert.storage.field_utils import field_utils


def test_that_egrid_files_are_saved_and_loaded_correctly(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=2)
        ensemble_dir = tmp_path / "ensembles" / str(ensemble.id)
        assert ensemble_dir.exists()

        grid = xtgeo.create_box_grid(dimension=(4, 5, 1))
        mask = grid.get_actnum()
        mask.values = [True] * 3 + [False] * 16 + [True]
        grid.set_actnum(mask)
        grid.to_file(f"{experiment.mount_point}/grid.EGRID", "egrid")

        data = [1.2, 1.1, 4.3, 3.1]
        ds = field_utils.create_field_dataset(
            ensemble.experiment.grid_path,
            data=data,
        )
        ensemble.save_parameters("MY_PARAM", 1, ds)
        saved_file = ensemble_dir / "realization-1" / "MY_PARAM.nc"
        assert saved_file.exists()

        loaded_data = xr.open_dataarray(saved_file).values
        expected_data = data[:-1] + [np.nan] * 16 + [data[3]]
        expected_data = np.asarray(expected_data)
        expected_data = expected_data.reshape(4, 5, 1)
        np.testing.assert_array_equal(loaded_data.squeeze(axis=0), expected_data)


def test_that_grid_files_are_saved_and_loaded_correctly(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=2)
        ensemble_dir = tmp_path / "ensembles" / str(ensemble.id)
        assert ensemble_dir.exists()

        mask = [True] * 3 + [False] * 16 + [True]
        grid = EclGridGenerator.create_rectangular((4, 5, 1), (1, 1, 1), actnum=mask)
        grid.save_GRID(f"{experiment.mount_point}/grid.GRID")

        data = [1.2, 1.1, 4.3, 3.1]
        ds = field_utils.create_field_dataset(
            ensemble.experiment.grid_path,
            data=data,
        )
        ensemble.save_parameters("MY_PARAM", 1, ds)

        saved_file = ensemble_dir / "realization-1" / "MY_PARAM.nc"
        assert saved_file.exists()

        loaded_data = xr.open_dataarray(saved_file).values
        expected_data = data[:-1] + [np.nan] * 16 + [data[3]]
        expected_data = np.asarray(expected_data)
        expected_data = expected_data.reshape(4, 5, 1, order="F")
        np.testing.assert_array_equal(loaded_data.squeeze(axis=0), expected_data)
