import polars as pl

from ert.storage.blob_data import BlobStorageData, ScalarsData
from ert.storage.migration.to43 import migrate


def test_that_migration_moves_scalar_parquet_into_blob(tmp_path):
    root = tmp_path / "project"
    ens_dir = root / "ensembles" / "ens-1"
    ens_dir.mkdir(parents=True)
    scalars = pl.DataFrame({"realization": [0, 1], "param": [1.5, 2.5]})
    scalars.write_parquet(ens_dir / "SCALAR.parquet")

    migrate(root)

    assert not (ens_dir / "SCALAR.parquet").exists()
    blob_dir = ens_dir / "blobs"
    [meta_file] = blob_dir.glob("*.blob.json")
    meta = BlobStorageData.model_validate_json(meta_file.read_text(encoding="utf-8"))
    assert isinstance(meta.blob_info, ScalarsData)
    assert meta.file_type == "application/parquet"
    assert meta.file_size == (blob_dir / meta.uri).stat().st_size
    assert pl.read_parquet(blob_dir / meta.uri).equals(scalars)


def test_that_migration_ignores_ensembles_without_scalar_parquet(tmp_path):
    root = tmp_path / "project"
    ens_dir = root / "ensembles" / "ens-1"
    (ens_dir / "realization-0").mkdir(parents=True)

    migrate(root)

    assert not (ens_dir / "blobs").exists()
