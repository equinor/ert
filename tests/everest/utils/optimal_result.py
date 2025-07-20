from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from everest.everest_storage import EverestStorage


@dataclass
class OptimalResult:
    batch: int
    controls: dict[str, Any]
    total_objective: float


def get_optimal_result(output_dir: str) -> OptimalResult | None:
    storage = EverestStorage.from_storage_path(Path(output_dir))
    storage.read_from_output_dir()

    matching_batches = [
        b
        for b in storage.batches_with_function_results
        if not b.batch_objectives.is_empty() and b.is_improvement
    ]

    if matching_batches:
        matching_batches.sort(
            key=lambda b: -b.batch_objectives.select(
                pl.col("total_objective_value").sample(n=1)
            ).item()
        )
        batch = matching_batches[0]
        controls_dict = batch.realization_controls.drop(
            [
                "batch_id",
                "simulation_id",
                "realization",
            ]
        ).to_dicts()[0]

        storage.close()

        return OptimalResult(
            batch=batch.batch_id,
            controls=controls_dict,
            total_objective=batch.batch_objectives.select(
                pl.col("total_objective_value")
            ).item(),
        )

    return None
