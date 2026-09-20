from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from everest.config.utils import CONSTRAINT_TOLERANCE, constraint_violation_check
from everest.util._utils import get_everest_experiment


@dataclass
class OptimalResult:
    batch: int
    controls: dict[str, Any]
    objectives: dict[str, Any]


def get_optimal_result(output_dir: str) -> OptimalResult | None:
    experiment = get_everest_experiment(Path(output_dir))
    max_total_objective = np.inf
    matching_batches = []
    for ens in experiment.ensembles_with_function_results:
        if ens.batch_objectives is None or ens.batch_objectives.is_empty():
            continue
        total_objective = ens.batch_objectives["total_objective_value"].item()

        bound_violation = constraint_violation_check(
            ens.batch_bound_constraint_violations
        )
        input_violation = constraint_violation_check(
            ens.batch_input_constraint_violations
        )
        output_violation = constraint_violation_check(
            ens.batch_output_constraint_violations
        )

        if (
            max(
                bound_violation,
                input_violation,
                output_violation,
            )
            < CONSTRAINT_TOLERANCE
            and total_objective < max_total_objective
        ):
            matching_batches.append(ens)
            max_total_objective = total_objective

    if matching_batches:
        matching_batches.sort(
            key=lambda item: item.batch_objectives.select(
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

        experiment._storage.close()

        objectives = {
            key: value[0]
            for key, value in batch.batch_objectives.drop(
                "batch_id", "total_objective_value"
            )
            .to_dict(as_series=False)
            .items()
        }
        return OptimalResult(
            batch=batch.iteration,
            controls=controls_dict,
            objectives=objectives,
        )

    experiment._storage.close()
    return None
