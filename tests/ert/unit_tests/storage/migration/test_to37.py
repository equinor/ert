from pathlib import Path

import polars as pl

from ert.storage.blob_data import BlobStorageData, EverestBatchData
from ert.storage.migration.to37 import migrate


def _write_batch_parquet(ens_dir: Path, dataframe_name: str) -> pl.DataFrame:
    df = pl.DataFrame({"batch_id": [0], dataframe_name: [1.5]})
    df.write_parquet(ens_dir / f"{dataframe_name}.parquet")
    return df


def test_that_migration_moves_batch_parquet_files_into_blobs(tmp_path):
    root = tmp_path / "project"
    ens_dir = root / "ensembles" / "ens-1"
    ens_dir.mkdir(parents=True)

    objectives = _write_batch_parquet(ens_dir, "batch_objectives")
    gradient = _write_batch_parquet(ens_dir, "batch_objective_gradient")

    migrate(root)

    assert not (ens_dir / "batch_objectives.parquet").exists()
    assert not (ens_dir / "batch_objective_gradient.parquet").exists()

    blob_dir = ens_dir / "blobs"
    metas = [
        BlobStorageData.model_validate_json(meta_file.read_text(encoding="utf-8"))
        for meta_file in sorted(blob_dir.glob("*.blob.json"))
    ]

    by_name = {meta.blob_info.dataframe_name: meta for meta in metas}
    assert set(by_name) == {"batch_objectives", "batch_objective_gradient"}

    for dataframe_name, expected in (
        ("batch_objectives", objectives),
        ("batch_objective_gradient", gradient),
    ):
        meta = by_name[dataframe_name]
        assert isinstance(meta.blob_info, EverestBatchData)
        assert meta.file_type == "application/parquet"
        assert meta.name == dataframe_name
        blob_bytes = (blob_dir / meta.uri).read_bytes()
        assert meta.file_size == len(blob_bytes)
        assert pl.read_parquet(blob_dir / meta.uri).equals(expected)


def test_that_migration_leaves_non_batch_parquet_files_untouched(tmp_path):
    root = tmp_path / "project"
    ens_dir = root / "ensembles" / "ens-1"
    realization_dir = ens_dir / "realization-0"
    realization_dir.mkdir(parents=True)

    responses = pl.DataFrame({"response_key": ["summary"], "values": [1.0]})
    responses.write_parquet(realization_dir / "responses.parquet")

    migrate(root)

    assert (realization_dir / "responses.parquet").exists()
    assert not (ens_dir / "blobs").exists()


def test_that_migration_is_a_noop_without_ensembles_directory(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    migrate(root)

    assert list(root.iterdir()) == []


def test_that_migration_records_only_present_batch_dataframes_as_blobs(tmp_path):
    root = tmp_path / "project"
    ens_dir = root / "ensembles" / "ens-1"
    ens_dir.mkdir(parents=True)

    _write_batch_parquet(ens_dir, "batch_constraints")

    migrate(root)

    blob_metas = list((ens_dir / "blobs").glob("*.blob.json"))
    assert len(blob_metas) == 1
    meta = BlobStorageData.model_validate_json(
        blob_metas[0].read_text(encoding="utf-8")
    )
    assert isinstance(meta.blob_info, EverestBatchData)
    assert meta.blob_info.dataframe_name == "batch_constraints"
