import io
import json

import numpy as np
import scipy as sp

from ert.analysis.event import AnalysisRhoMatrixEvent
from ert.storage import open_storage
from ert.storage.blob_data import BlobType, RhoStorageData


def _make_rho_event(
    param_name: str = "FIELD_A",
    shape: tuple[int, int] = (6, 2),
    observation_keys: list[str] | None = None,
) -> tuple[AnalysisRhoMatrixEvent, np.ndarray]:
    rng = np.random.default_rng(42)
    dense = rng.random(shape).astype(np.float64)
    dense[dense < 0.5] = 0.0
    sparse = sp.sparse.csc_matrix(dense)
    buf = io.BytesIO()
    sp.sparse.save_npz(buf, sparse)
    event = AnalysisRhoMatrixEvent(
        param_name=param_name,
        observation_keys=observation_keys or ["OBS_1", "OBS_2"],
        shape=shape,
        data_type=str(dense.dtype),
        matrix_bytes=buf.getvalue(),
    )
    return event, dense


def test_that_load_rho_matrix_returns_none_when_no_blob_exists(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()

        assert experiment.load_rho_matrix("NONEXISTENT") is None


def test_that_rho_matrix_metadata_contains_observation_keys(tmp_path):
    obs_keys = ["WOPR:OP1", "WGPR:OP2", "FOPR"]
    event, _ = _make_rho_event(observation_keys=obs_keys, shape=(6, 3))

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        blobs = experiment._load_blob_metadata(BlobType.RHO_MATRIX)

    assert len(blobs) == 1
    blob = blobs[0]
    assert isinstance(blob.blob_info, RhoStorageData)
    assert blob.blob_info.param_name == "FIELD_A"
    assert blob.blob_info.observation_keys == obs_keys
    assert blob.blob_info.sparse is True
    assert blob.blob_info.shape == (6, 3)
    assert blob.file_type == "application/x-npz"


def test_that_rho_matrix_blob_files_are_written_to_disk(tmp_path):
    event, _ = _make_rho_event()

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        blob_dir = experiment._path / "blobs"
        assert blob_dir.is_dir()

        blob_files = list(blob_dir.glob("*.blob"))
        json_files = list(blob_dir.glob("*.blob.json"))
        assert len(blob_files) == 1
        assert len(json_files) == 1

        meta = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert meta["blob_info"]["blob_type"] == "rho_matrix"
        assert meta["blob_info"]["param_name"] == "FIELD_A"
        assert meta["file_size"] > 0


def test_that_load_rho_matrix_distinguishes_parameters_by_name(tmp_path):
    event_a, dense_a = _make_rho_event(
        param_name="PORO", shape=(4, 3), observation_keys=["O1", "O2", "O3"]
    )
    event_b, dense_b = _make_rho_event(
        param_name="PERM", shape=(5, 2), observation_keys=["O4", "O5"]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event_a)
        experiment.save_blob(event_b)

        loaded_a = experiment.load_rho_matrix("PORO")
        loaded_b = experiment.load_rho_matrix("PERM")

    assert loaded_a is not None
    assert loaded_b is not None
    np.testing.assert_array_equal(loaded_a, dense_a)
    np.testing.assert_array_equal(loaded_b, dense_b)


def test_that_load_rho_matrix_validates_observation_keys(tmp_path):
    """Cached rho is valid for subset/equal keys, stale when keys are missing."""
    stored_keys = ["OBS_A", "OBS_B", "OBS_C"]
    event, dense = _make_rho_event(
        param_name="PORO", shape=(4, 3), observation_keys=stored_keys
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        # full set of observation keys — should return cached matrix
        result = experiment.load_rho_matrix("PORO", observation_keys=stored_keys)
        assert result is not None
        np.testing.assert_array_equal(result, dense)

        # subset of observation keys — should return those columns, and only those
        result = experiment.load_rho_matrix("PORO", observation_keys=["OBS_A", "OBS_C"])
        assert result is not None
        np.testing.assert_array_equal(result, dense[:, [0, 2]])

        # no observation keys specified — should return cached matrix
        result = experiment.load_rho_matrix("PORO", observation_keys=None)
        assert result is not None
        np.testing.assert_array_equal(result, dense)

        # missing observation keys — should be treated as invalid and return None
        result = experiment.load_rho_matrix(
            "PORO", observation_keys=["OBS_A", "OBS_NEW"]
        )
        assert result is None


def test_that_load_rho_matrix_is_cut_down_to_the_active_observations(tmp_path):
    """One column per active observation, in the order they are assimilated.

    Observations are deactivated between the assimilations of an ES-MDA, so a
    rho matrix cached during the first one describes more observations than the
    later ones use. It is multiplied elementwise with a Kalman gain that has one
    column per active observation, so the columns have to line up.
    """
    event, dense = _make_rho_event(
        param_name="PORO", shape=(4, 3), observation_keys=["OBS_A", "OBS_B", "OBS_C"]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        for active, columns in [
            (["OBS_A", "OBS_B", "OBS_C"], [0, 1, 2]),
            (["OBS_B", "OBS_C"], [1, 2]),
            (["OBS_B"], [1]),
            (["OBS_C", "OBS_A"], [2, 0]),
        ]:
            result = experiment.load_rho_matrix("PORO", observation_keys=active)
            assert result is not None
            assert result.shape == (4, len(active))
            np.testing.assert_array_equal(result, dense[:, columns])


def test_that_load_rho_matrix_is_recomputed_when_stored_keys_are_not_unique(tmp_path):
    """A repeated key gives no way to tell its columns apart, so do not guess.

    An observation of several indices contributes one key per index, each with
    its own position and so its own column. Subsetting by key would pick columns
    arbitrarily, which is worse than recomputing, so None is returned.
    """
    event, dense = _make_rho_event(
        param_name="PORO", shape=(4, 3), observation_keys=["OBS_A", "OBS_A", "OBS_B"]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        # The stored set, unchanged, is still served: the columns line up as they are
        result = experiment.load_rho_matrix(
            "PORO", observation_keys=["OBS_A", "OBS_A", "OBS_B"]
        )
        assert result is not None
        np.testing.assert_array_equal(result, dense)

        # Anything else is refused rather than guessed at
        assert experiment.load_rho_matrix("PORO", observation_keys=["OBS_A"]) is None
        assert (
            experiment.load_rho_matrix("PORO", observation_keys=["OBS_A", "OBS_B"])
            is None
        )
