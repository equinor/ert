import io
from pathlib import Path

import polars as pl

from ert.storage.blob_data import BlobStorageData, ScalingFactorsData
from ert.storage.migration.to32 import migrate


def _read_blob_metadata(blob_meta_path: Path) -> BlobStorageData:
    return BlobStorageData.model_validate_json(
        blob_meta_path.read_text(encoding="utf-8")
    )


def test_that_migration_moves_scaling_factor_parquet_to_blob_storage(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    ensemble_dir = root / "ensembles" / "ens-1"
    ensemble_dir.mkdir(parents=True)

    old_path = ensemble_dir / "observation_scaling_factors.parquet"
    old_df = pl.DataFrame(
        {
            "input_group": ["obs1*", "obs2*"],
            "index": ["r0", "r1"],
            "obs_key": ["OBS_1", "OBS_2"],
            "scaling_factor": pl.Series([2.0, 3.0], dtype=pl.Float32),
        }
    )
    old_df.write_parquet(old_path)

    migrate(root)

    assert not old_path.exists()

    blob_dir = ensemble_dir / "blobs"
    blob_data_files = list(blob_dir.glob("*.blob"))
    blob_meta_files = list(blob_dir.glob("*.blob.json"))
    assert len(blob_data_files) == 1
    assert len(blob_meta_files) == 1

    meta = _read_blob_metadata(blob_meta_files[0])
    assert meta.name == "scaling_factors"
    assert meta.file_type == "application/parquet"
    assert isinstance(meta.blob_info, ScalingFactorsData)
    assert meta.blob_info.update_algorithm == "ensemble_smoother"
    assert meta.blob_info.num_observations == 2
    assert meta.blob_info.num_groups == 2

    migrated_df = pl.read_parquet(io.BytesIO((blob_dir / meta.uri).read_bytes()))
    assert migrated_df.to_dict(as_series=False) == old_df.to_dict(as_series=False)


def test_that_migration_defaults_num_groups_to_one_when_input_group_is_missing(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()

    ensemble_dir = root / "ensembles" / "ens-1"
    ensemble_dir.mkdir(parents=True)

    old_path = ensemble_dir / "observation_scaling_factors.parquet"
    pl.DataFrame(
        {
            "index": ["r0", "r1"],
            "obs_key": ["OBS_1", "OBS_2"],
            "scaling_factor": pl.Series([2.0, 3.0], dtype=pl.Float32),
        }
    ).write_parquet(old_path)

    migrate(root)

    blob_meta_files = list((ensemble_dir / "blobs").glob("*.blob.json"))
    assert len(blob_meta_files) == 1

    meta = _read_blob_metadata(blob_meta_files[0])
    assert isinstance(meta.blob_info, ScalingFactorsData)
    assert meta.blob_info.num_observations == 2
    assert meta.blob_info.num_groups == 1
