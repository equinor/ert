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
    event, _ = _make_rho_event(observation_keys=obs_keys)

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        experiment.save_blob(event)

        blobs = experiment.load_blobs(BlobType.RHO_MATRIX)

    assert len(blobs) == 1
    blob = blobs[0]
    assert isinstance(blob.blob_info, RhoStorageData)
    assert blob.blob_info.param_name == "FIELD_A"
    assert blob.blob_info.observation_keys == obs_keys
    assert blob.blob_info.sparse is True
    assert blob.blob_info.shape == (6, 2)
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
    event_a, dense_a = _make_rho_event(param_name="PORO", shape=(4, 3))
    event_b, dense_b = _make_rho_event(param_name="PERM", shape=(5, 2))

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
