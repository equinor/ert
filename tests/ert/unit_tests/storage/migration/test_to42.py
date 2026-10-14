import json
from pathlib import Path

from ert.storage.migration.to42 import migrate


def _write_blob(blob_dir: Path, uri: str, blob_type: str, name: str) -> None:
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / uri).write_bytes(b"fake-bytes")
    (blob_dir / f"{uri}.json").write_text(
        json.dumps(
            {
                "uri": uri,
                "file_size": 10,
                "file_type": "application/x-npz",
                "name": name,
                "blob_info": {
                    "blob_type": blob_type,
                    "update_algorithm": "distance",
                    "data_type": "float32",
                    "param_name": name,
                    "observation_keys": ["SEISMIC", "SEISMIC"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_that_migration_removes_rho_matrix_blobs(tmp_path):
    blob_dir = tmp_path / "experiments" / "exp-1" / "blobs"
    _write_blob(blob_dir, "aaaaaaaa.blob", "rho_matrix", "PORO")

    migrate(tmp_path)

    assert not (blob_dir / "aaaaaaaa.blob").exists()
    assert not (blob_dir / "aaaaaaaa.blob.json").exists()


def test_that_migration_leaves_other_blob_types_untouched(tmp_path):
    blob_dir = tmp_path / "experiments" / "exp-1" / "blobs"
    _write_blob(blob_dir, "bbbbbbbb.blob", "matrix", "K")

    migrate(tmp_path)

    assert (blob_dir / "bbbbbbbb.blob").exists()
    assert (blob_dir / "bbbbbbbb.blob.json").exists()


def test_that_migration_will_not_delete_rho_matrix_files_outside_blob_directory(
    tmp_path,
):
    exp_dir = tmp_path / "experiments" / "exp-1"
    blob_dir = exp_dir / "blobs"
    blob_dir.mkdir(parents=True)
    outside = exp_dir / "outside.blob"
    outside.write_bytes(b"outside")
    (blob_dir / "evil.blob.json").write_text(
        json.dumps(
            {
                "uri": "../outside.blob",
                "file_size": 7,
                "file_type": "application/x-npz",
                "name": "PORO",
                "blob_info": {"blob_type": "rho_matrix"},
            }
        ),
        encoding="utf-8",
    )

    migrate(tmp_path)

    assert outside.exists()
