from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def experiment_dirs(storage_path: Path) -> Iterator[Path]:
    experiments_dir = storage_path / "experiments"
    if experiments_dir.exists():
        yield from (d for d in experiments_dir.iterdir() if d.is_dir())


def ensemble_dirs(storage_path: Path) -> Iterator[Path]:
    ensembles_dir = storage_path / "ensembles"
    if ensembles_dir.exists():
        yield from (d for d in ensembles_dir.iterdir() if d.is_dir())


def experiment_index_files(storage_path: Path) -> Iterator[Path]:
    for experiment_dir in experiment_dirs(storage_path):
        index_file = experiment_dir / "index.json"
        if index_file.exists():
            yield index_file


def get_ensemble_index_data(ensemble_dir: Path) -> dict[str, Any]:
    return read_json(ensemble_dir / "index.json")


def resolve_blob_path(blob_dir: Path, uri: str) -> Path | None:
    """Resolve a blob uri, or return None if it points outside of blob_dir."""
    resolved_blob_dir = blob_dir.resolve()
    blob_path = (resolved_blob_dir / uri).resolve()
    if not blob_path.is_relative_to(resolved_blob_dir):
        return None
    return blob_path


def write_parquet_blob(
    ensemble_dir: Path, data: bytes, name: str, blob_info: dict[str, Any]
) -> str:
    """Write parquet bytes with blob metadata into the ensemble blob directory.

    Returns:
        The uri of the written blob.
    """
    blob_dir = ensemble_dir / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    uri = f"{uuid.uuid4().hex[:8]}.blob"
    (blob_dir / uri).write_bytes(data)
    write_json(
        blob_dir / f"{uri}.json",
        {
            "uri": uri,
            "file_size": len(data),
            "file_type": "application/parquet",
            "name": name,
            "blob_info": blob_info,
        },
    )
    return uri
