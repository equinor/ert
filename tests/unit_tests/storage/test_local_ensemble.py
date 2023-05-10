import numpy
import xtgeo
from ecl.grid import EclGridGenerator

from ert.storage import open_storage


def test_save_field_xtgeo(tmp_path):
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
        ensemble.save_field(
            parameter_name="MY_PARAM",
            realization=1,
            data=data,
        )

        saved_file = ensemble_dir / "realization-1" / "MY_PARAM.npy"
        assert saved_file.exists()

        loaded_data = numpy.load(saved_file)
        expected_data = data[:-1] + [numpy.nan] * 16 + [data[3]]
        expected_data = numpy.asarray(expected_data)
        expected_data = expected_data.reshape(4, 5, 1)
        numpy.testing.assert_array_equal(loaded_data, expected_data)


def test_save_field_ecl(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=2)
        ensemble_dir = tmp_path / "ensembles" / str(ensemble.id)
        assert ensemble_dir.exists()

        mask = [True] * 3 + [False] * 16 + [True]
        grid = EclGridGenerator.create_rectangular((4, 5, 1), (1, 1, 1), actnum=mask)
        grid.save_GRID(f"{experiment.mount_point}/grid.GRID")

        data = [1.2, 1.1, 4.3, 3.1]
        ensemble.save_field(
            parameter_name="MY_PARAM",
            realization=1,
            data=data,
        )

        saved_file = ensemble_dir / "realization-1" / "MY_PARAM.npy"
        assert saved_file.exists()

        loaded_data = numpy.load(saved_file)
        expected_data = data[:-1] + [numpy.nan] * 16 + [data[3]]
        expected_data = numpy.asarray(expected_data)
        expected_data = expected_data.reshape(4, 5, 1, order="F")
        numpy.testing.assert_array_equal(loaded_data, expected_data)
