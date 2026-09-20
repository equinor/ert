import json
from pathlib import Path

from ert.storage.blob_data import BlobStorageData, MatrixStorageData
from ert.storage.migration.to34 import migrate


def _write_blob_meta(blob_dir: Path, blob_id: str, meta: dict) -> Path:
    meta_file = blob_dir / f"{blob_id}.blob.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_file


def test_that_migration_adds_parameter_group_sizes_only_to_matrix_blobs(tmp_path):
    root = tmp_path / "project"
    blob_dir = root / "ensembles" / "ens-1" / "blobs"
    blob_dir.mkdir(parents=True)

    matrix_meta = {
        "uri": "abc123.blob",
        "file_size": 1024,
        "file_type": "application/x-npy",
        "name": "H",
        "blob_info": {
            "blob_type": "matrix",
            "update_algorithm": "enif",
            "sparse": False,
            "shape": [10, 5],
            "data_type": "float64",
        },
    }
    _write_blob_meta(blob_dir, "abc123", matrix_meta)

    obs_meta = {
        "uri": "obs789.blob",
        "file_size": 512,
        "file_type": "application/parquet",
        "name": "observation_report",
        "blob_info": {
            "blob_type": "observation_report",
            "update_algorithm": "ensemble_smoother",
        },
    }
    obs_meta_file = _write_blob_meta(blob_dir, "obs789", obs_meta)

    migrate(root)

    updated_matrix = BlobStorageData.model_validate_json(
        (blob_dir / "abc123.blob.json").read_text(encoding="utf-8")
    )
    assert isinstance(updated_matrix.blob_info, MatrixStorageData)
    assert updated_matrix.blob_info.parameter_group_sizes == {}

    updated_obs = json.loads(obs_meta_file.read_text(encoding="utf-8"))
    assert "parameter_group_sizes" not in updated_obs["blob_info"]
