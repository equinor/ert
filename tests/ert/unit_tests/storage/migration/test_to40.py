import json
from pathlib import Path

from ert.storage.migration.to40 import migrate


def _write_blob(
    blob_dir: Path,
    uri: str,
    name: str,
    update_algorithm: str,
    blob_type: str = "matrix",
) -> None:
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / uri).write_bytes(b"fake-matrix-bytes")
    (blob_dir / f"{uri}.json").write_text(
        json.dumps(
            {
                "uri": uri,
                "file_size": 18,
                "file_type": "application/x-npy",
                "name": name,
                "blob_info": {
                    "blob_type": blob_type,
                    "update_algorithm": update_algorithm,
                    "data_type": "float64",
                },
            }
        ),
        encoding="utf-8",
    )


def test_that_migration_removes_enif_gain_blobs(tmp_path):
    root = tmp_path / "project"
    blob_dir = root / "ensembles" / "ens-1" / "blobs"
    _write_blob(blob_dir, "aaaaaaaa.blob", "K", "enif")

    migrate(root)

    assert not (blob_dir / "aaaaaaaa.blob").exists()
    assert not (blob_dir / "aaaaaaaa.blob.json").exists()


def test_that_migration_leaves_other_enif_blobs_untouched(tmp_path):
    root = tmp_path / "project"
    blob_dir = root / "ensembles" / "ens-1" / "blobs"
    _write_blob(blob_dir, "bbbbbbbb.blob", "H", "enif")
    _write_blob(blob_dir, "cccccccc.blob", "Prec_u", "enif")

    migrate(root)

    assert (blob_dir / "bbbbbbbb.blob").exists()
    assert (blob_dir / "bbbbbbbb.blob.json").exists()
    assert (blob_dir / "cccccccc.blob").exists()
    assert (blob_dir / "cccccccc.blob.json").exists()
