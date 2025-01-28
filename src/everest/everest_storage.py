from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    TypedDict,
    cast,
)

import numpy as np
import polars
from ropt.enums import EventType
from ropt.plan import BasicOptimizer, Event
from ropt.results import FunctionResults, GradientResults, convert_to_maximize

from everest.config import EverestConfig
from everest.strings import EVEREST

logger = logging.getLogger(__name__)


@dataclass
class OptimalResult:
    batch: int
    controls: list[Any]
    total_objective: float


def try_read_df(path: Path) -> polars.DataFrame | None:
    return polars.read_parquet(path) if path.exists() else None


@dataclass
class BatchStorageData:
    batch_id: int
    realization_controls: polars.DataFrame
    batch_objectives: polars.DataFrame | None
    realization_objectives: polars.DataFrame | None
    batch_constraints: polars.DataFrame | None
    realization_constraints: polars.DataFrame | None
    batch_objective_gradient: polars.DataFrame | None
    perturbation_objectives: polars.DataFrame | None
    batch_constraint_gradient: polars.DataFrame | None
    perturbation_constraints: polars.DataFrame | None
    is_improvement: bool | None = False

    @property
    def existing_dataframes(self) -> dict[str, polars.DataFrame]:
        return {
            k: cast(polars.DataFrame, getattr(self, k))
            for k in [
                "batch_objectives",
                "batch_objective_gradient",
                "batch_constraints",
                "batch_constraint_gradient",
                "realization_controls",
                "realization_objectives",
                "realization_constraints",
                "perturbation_objectives",
                "perturbation_constraints",
            ]
            if getattr(self, k) is not None
        }


@dataclass
class OptimizationStorageData:
    batches: list[BatchStorageData] = field(default_factory=list)
    controls: polars.DataFrame | None = None
    objective_functions: polars.DataFrame | None = None
    nonlinear_constraints: polars.DataFrame | None = None
    realization_weights: polars.DataFrame | None = None

    def simulation_to_geo_realization_map(self, batch_id: int) -> dict[int, int]:
        """
        Mapping from simulation ID to geo-realization
        """
        dummy_df = next(
            (
                b.realization_controls
                for b in self.batches
                if b.batch_id == batch_id and b.realization_controls is not None
            ),
            None,
        )

        if dummy_df is None:
            return {}

        mapping = {}
        for d in dummy_df.select("realization", "simulation_id").to_dicts():
            mapping[int(d["simulation_id"])] = int(d["realization"])

        return mapping

    @property
    def existing_dataframes(self) -> dict[str, polars.DataFrame]:
        return {
            k: cast(polars.DataFrame, getattr(self, k))
            for k in [
                "controls",
                "objective_functions",
                "nonlinear_constraints",
                "realization_weights",
            ]
            if getattr(self, k) is not None
        }

    def write_to_experiment(self, experiment: _OptimizerOnlyExperiment) -> None:
        for df_name, df in self.existing_dataframes.items():
            df.write_parquet(f"{experiment.optimizer_mount_point / df_name}.parquet")

        for batch_data in self.batches:
            ensemble = experiment.get_ensemble_by_name(f"batch_{batch_data.batch_id}")
            with open(
                ensemble.optimizer_mount_point / "batch.json", "w+", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "batch_id": batch_data.batch_id,
                        "is_improvement": batch_data.is_improvement,
                    },
                    f,
                )

            for df_key, df in batch_data.existing_dataframes.items():
                df.write_parquet(ensemble.optimizer_mount_point / f"{df_key}.parquet")

    def read_from_experiment(self, experiment: _OptimizerOnlyExperiment) -> None:
        self.controls = polars.read_parquet(
            experiment.optimizer_mount_point / "controls.parquet"
        )
        self.objective_functions = polars.read_parquet(
            experiment.optimizer_mount_point / "objective_functions.parquet"
        )

        if (
            experiment.optimizer_mount_point / "nonlinear_constraints.parquet"
        ).exists():
            self.nonlinear_constraints = polars.read_parquet(
                experiment.optimizer_mount_point / "nonlinear_constraints.parquet"
            )

        if (experiment.optimizer_mount_point / "realization_weights.parquet").exists():
            self.realization_weights = polars.read_parquet(
                experiment.optimizer_mount_point / "realization_weights.parquet"
            )

        for ens in experiment.ensembles.values():
            with open(ens.optimizer_mount_point / "batch.json", encoding="utf-8") as f:
                info = json.load(f)

            self.batches.append(
                BatchStorageData(
                    batch_id=info["batch_id"],
                    **{
                        df_name: try_read_df(
                            Path(ens.optimizer_mount_point) / f"{df_name}.parquet"
                        )
                        for df_name in [
                            "batch_objectives",
                            "batch_objective_gradient",
                            "batch_constraints",
                            "batch_constraint_gradient",
                            "realization_controls",
                            "realization_objectives",
                            "realization_constraints",
                            "perturbation_objectives",
                            "perturbation_constraints",
                        ]
                    },
                    is_improvement=info["is_improvement"],
                )
            )

        self.batches.sort(key=lambda b: b.batch_id)


class _OptimizerOnlyEnsemble:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def optimizer_mount_point(self) -> Path:
        if not (self._output_dir / "optimizer").exists():
            Path.mkdir(self._output_dir / "optimizer", parents=True)

        return self._output_dir / "optimizer"


class _OptimizerOnlyExperiment:
    """
    Mocks an ERT storage, if we want to store optimization results within the
    ERT storage, we can use an ERT Experiment object with an optimizer_mount_point
    property
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._ensembles = {}

    @property
    def optimizer_mount_point(self) -> Path:
        if not (self._output_dir / "optimizer").exists():
            Path.mkdir(self._output_dir / "optimizer", parents=True)

        return self._output_dir / "optimizer"

    @property
    def ensembles(self) -> dict[str, _OptimizerOnlyEnsemble]:
        return {
            str(d): _OptimizerOnlyEnsemble(self._output_dir / "ensembles" / d)
            for d in os.listdir(self._output_dir / "ensembles")
            if "batch_" in d
        }

    def get_ensemble_by_name(self, name: str) -> _OptimizerOnlyEnsemble:
        if name not in self._ensembles:
            self._ensembles[name] = _OptimizerOnlyEnsemble(
                self._output_dir / "ensembles" / name
            )

        return self._ensembles[name]


@dataclass
class _EvaluationResults(TypedDict):
    realization_controls: polars.DataFrame
    batch_objectives: polars.DataFrame
    realization_objectives: polars.DataFrame
    batch_constraints: polars.DataFrame | None
    realization_constraints: polars.DataFrame | None


@dataclass
class _GradientResults(TypedDict):
    batch_objective_gradient: polars.DataFrame | None
    perturbation_objectives: polars.DataFrame | None
    batch_constraint_gradient: polars.DataFrame | None
    perturbation_constraints: polars.DataFrame | None


@dataclass
class _MeritValue:
    value: float
    iter: int


class EverestStorage:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        self._control_ensemble_id = 0
        self._gradient_ensemble_id = 0

        self._output_dir = output_dir
        self.data = OptimizationStorageData()

    @staticmethod
    def _rename_ropt_df_columns(df: polars.DataFrame) -> polars.DataFrame:
        """
        Renames columns of a dataframe from ROPT to what will be displayed
        to the user.
        """
        scaled_cols = [c for c in df.columns if c.lower().startswith("scaled")]
        if len(scaled_cols) > 0:
            raise ValueError("Scaled columns should not be stored into Everest storage")

        # Keys are ROPT column keys
        # values are corresponding column keys we present to the user
        renames = {
            "objective": "objective_name",
            "weighted_objective": "total_objective_value",
            "variable": "control_name",
            "variables": "control_value",
            "objectives": "objective_value",
            "constraints": "constraint_value",
            "nonlinear_constraint": "constraint_name",
            "perturbed_variables": "perturbed_control_value",
            "perturbed_objectives": "perturbed_objective_value",
            "perturbed_constraints": "perturbed_constraint_value",
            "evaluation_ids": "simulation_id",
        }
        return df.rename({k: v for k, v in renames.items() if k in df.columns})

    @staticmethod
    def _enforce_dtypes(df: polars.DataFrame) -> polars.DataFrame:
        dtypes = {
            "batch_id": polars.UInt32,
            "perturbation": polars.UInt32,
            "realization": polars.UInt32,
            # -1 is used as a value in simulator cache.
            # thus we need signed, otherwise we could do unsigned
            "simulation_id": polars.Int32,
            "perturbed_evaluation_ids": polars.Int32,
            "objective_name": polars.String,
            "control_name": polars.String,
            "constraint_name": polars.String,
            "total_objective_value": polars.Float64,
            "control_value": polars.Float64,
            "objective_value": polars.Float64,
            "constraint_value": polars.Float64,
            "perturbed_control_value": polars.Float64,
            "perturbed_objective_value": polars.Float64,
            "perturbed_constraint_value": polars.Float64,
        }

        existing_cols = set(df.columns)
        unaccounted_cols = existing_cols - set(dtypes)
        if len(unaccounted_cols) > 0:
            raise KeyError(
                f"Expected all keys to have a specified dtype, found {unaccounted_cols}"
            )

        df = df.cast(
            {
                colname: dtype
                for colname, dtype in dtypes.items()
                if colname in df.columns
            }
        )

        return df

    def write_to_output_dir(self) -> None:
        exp = _OptimizerOnlyExperiment(self._output_dir)

        # csv writing mostly for dev/debugging/quick inspection
        self.data.write_to_experiment(exp)

    @staticmethod
    def check_for_deprecated_seba_storage(config_file: str):
        config = EverestConfig.load_file(config_file)
        output_dir = Path(config.optimization_output_dir)
        if os.path.exists(output_dir / "seba.db") or os.path.exists(
            output_dir / "seba.db.backup"
        ):
            trace = "\n".join(traceback.format_stack())
            logging.getLogger(EVEREST).error(
                f"Tried opening old seba storage.Traceback: {trace}"
            )
            raise SystemExit(
                f"Trying to open old storage @ {output_dir}/seba.db."
                f"This storage can only be opened with an ert[everest] version <= 12.1.2"
            )

    def read_from_output_dir(self) -> None:
        exp = _OptimizerOnlyExperiment(self._output_dir)
        self.data.read_from_experiment(exp)

    def observe_optimizer(
        self,
        optimizer: BasicOptimizer,
    ) -> None:
        optimizer.add_observer(
            EventType.START_OPTIMIZER_STEP, self._on_start_optimization
        )
        optimizer.add_observer(
            EventType.FINISHED_EVALUATION, self._on_batch_evaluation_finished
        )
        optimizer.add_observer(
            EventType.FINISHED_OPTIMIZER_STEP,
            self._on_optimization_finished,
        )

    def _on_start_optimization(self, event: Event) -> None:
        def _format_control_names(control_names):
            converted_names = []
            for name in control_names:
                converted = f"{name[0]}_{name[1]}"
                if len(name) > 2:
                    converted += f"-{name[2]}"
                converted_names.append(converted)

            return converted_names

        config = event.config

        # Note: We probably do not have to store
        # all of this information, consider removing.
        self.data.controls = polars.DataFrame(
            {
                "control_name": polars.Series(
                    _format_control_names(config.variables.names), dtype=polars.String
                ),
                "initial_value": polars.Series(
                    config.variables.initial_values, dtype=polars.Float64
                ),
                "lower_bounds": polars.Series(
                    config.variables.lower_bounds, dtype=polars.Float64
                ),
                "upper_bounds": polars.Series(
                    config.variables.upper_bounds, dtype=polars.Float64
                ),
            }
        )

        self.data.objective_functions = polars.DataFrame(
            {
                "objective_name": config.objectives.names,
                "weight": polars.Series(
                    config.objectives.weights, dtype=polars.Float64
                ),
                "normalization": polars.Series(  # Q: Is this correct?
                    [1.0 / s for s in config.objectives.scales],
                    dtype=polars.Float64,
                ),
            }
        )

        if config.nonlinear_constraints is not None:
            self.data.nonlinear_constraints = polars.DataFrame(
                {
                    "constraint_name": config.nonlinear_constraints.names,
                    "normalization": [
                        1.0 / s for s in config.nonlinear_constraints.scales
                    ],  # Q: Is this correct?
                    "constraint_rhs_value": config.nonlinear_constraints.rhs_values,
                    "constraint_type": config.nonlinear_constraints.types,
                }
            )

        self.data.realization_weights = polars.DataFrame(
            {
                "realization": polars.Series(
                    config.realizations.names, dtype=polars.UInt32
                ),
                "weight": polars.Series(
                    config.realizations.weights, dtype=polars.Float64
                ),
            }
        )

    def _store_function_results(self, results: FunctionResults) -> _EvaluationResults:
        # We could select only objective values,
        # but we select all to also get the constraint values (if they exist)
        realization_objectives = polars.from_pandas(
            results.to_dataframe(
                "evaluations",
                select=["objectives", "evaluation_ids"],
            ).reset_index(),
        ).select(
            "batch_id",
            "realization",
            "objective",
            "objectives",
            "evaluation_ids",
        )

        if results.functions is not None and results.functions.constraints is not None:
            realization_constraints = polars.from_pandas(
                results.to_dataframe(
                    "evaluations",
                    select=["constraints", "evaluation_ids"],
                ).reset_index(),
            ).select(
                "batch_id",
                "realization",
                "evaluation_ids",
                "nonlinear_constraint",
                "constraints",
            )

            realization_constraints = self._rename_ropt_df_columns(
                realization_constraints
            )

            batch_constraints = polars.from_pandas(
                results.to_dataframe("functions", select=["constraints"]).reset_index()
            ).select("batch_id", "nonlinear_constraint", "constraints")

            batch_constraints = batch_constraints.rename(
                {
                    "nonlinear_constraint": "constraint_name",
                    "constraints": "constraint_value",
                }
            )

            batch_constraints = batch_constraints.pivot(
                on="constraint_name",
                values=[
                    "constraint_value",
                ],
            )

            realization_constraints = realization_constraints.pivot(
                values=["constraint_value"], on="constraint_name"
            )
        else:
            batch_constraints = None
            realization_constraints = None

        batch_objectives = polars.from_pandas(
            results.to_dataframe(
                "functions",
                select=["objectives", "weighted_objective"],
            ).reset_index()
        ).select("batch_id", "objective", "objectives", "weighted_objective")

        realization_controls = polars.from_pandas(
            results.to_dataframe(
                "evaluations", select=["variables", "evaluation_ids"]
            ).reset_index()
        ).select(
            "batch_id",
            "variable",
            "realization",
            "variables",
            "evaluation_ids",
        )

        realization_controls = self._rename_ropt_df_columns(realization_controls)
        realization_controls = self._enforce_dtypes(realization_controls)

        realization_controls = realization_controls.pivot(
            on="control_name",
            values=["control_value"],
            separator=":",
        )

        batch_objectives = self._rename_ropt_df_columns(batch_objectives)
        batch_objectives = self._enforce_dtypes(batch_objectives)

        realization_objectives = self._rename_ropt_df_columns(realization_objectives)
        realization_objectives = self._enforce_dtypes(realization_objectives)

        batch_objectives = batch_objectives.pivot(
            on="objective_name",
            values=["objective_value"],
            separator=":",
        )

        realization_objectives = realization_objectives.pivot(
            values="objective_value",
            index=[
                "batch_id",
                "realization",
                "simulation_id",
            ],
            columns="objective_name",
        )

        return {
            "realization_controls": realization_controls,
            "batch_objectives": batch_objectives,
            "realization_objectives": realization_objectives,
            "batch_constraints": batch_constraints,
            "realization_constraints": realization_constraints,
        }

    def _store_gradient_results(self, results: GradientResults) -> _GradientResults:
        perturbation_objectives = polars.from_pandas(
            results.to_dataframe("evaluations").reset_index()
        ).select(
            [
                "batch_id",
                "variable",
                "realization",
                "perturbation",
                "objective",
                "variables",
                "perturbed_variables",
                "perturbed_objectives",
                "perturbed_evaluation_ids",
                *(
                    ["nonlinear_constraint", "perturbed_constraints"]
                    if results.evaluations.perturbed_constraints is not None
                    else []
                ),
            ]
        )
        perturbation_objectives = self._rename_ropt_df_columns(perturbation_objectives)
        perturbation_objectives = self._enforce_dtypes(perturbation_objectives)

        if results.gradients is not None:
            batch_objective_gradient = polars.from_pandas(
                results.to_dataframe("gradients").reset_index()
            ).select(
                [
                    "batch_id",
                    "variable",
                    "objective",
                    "weighted_objective",
                    "objectives",
                    *(
                        ["nonlinear_constraint", "constraints"]
                        if results.gradients.constraints is not None
                        else []
                    ),
                ]
            )
            batch_objective_gradient = self._rename_ropt_df_columns(
                batch_objective_gradient
            )
            batch_objective_gradient = self._enforce_dtypes(batch_objective_gradient)
        else:
            batch_objective_gradient = None

        if results.evaluations.perturbed_constraints is not None:
            perturbation_constraints = (
                perturbation_objectives[
                    "batch_id",
                    "realization",
                    "perturbation",
                    "control_name",
                    "perturbed_control_value",
                    *[
                        c
                        for c in perturbation_objectives.columns
                        if "constraint" in c.lower()
                    ],
                ]
                .pivot(on="constraint_name", values=["perturbed_constraint_value"])
                .pivot(on="control_name", values="perturbed_control_value")
            )

            if batch_objective_gradient is not None:
                batch_constraint_gradient = batch_objective_gradient[
                    "batch_id",
                    "control_name",
                    *[
                        c
                        for c in batch_objective_gradient.columns
                        if "constraint" in c.lower()
                    ],
                ]

                batch_objective_gradient = batch_objective_gradient.drop(
                    [
                        c
                        for c in batch_objective_gradient.columns
                        if "constraint" in c.lower()
                    ]
                ).unique()

                batch_constraint_gradient = batch_constraint_gradient.pivot(
                    on="constraint_name",
                    values=["constraint_value"],
                )
            else:
                batch_constraint_gradient = None

            perturbation_objectives = perturbation_objectives.drop(
                [
                    c
                    for c in perturbation_objectives.columns
                    if "constraint" in c.lower()
                ]
            ).unique()
        else:
            batch_constraint_gradient = None
            perturbation_constraints = None

        perturbation_objectives = perturbation_objectives.drop(
            "perturbed_evaluation_ids", "control_value"
        )

        perturbation_objectives = perturbation_objectives.pivot(
            on="objective_name", values="perturbed_objective_value"
        )

        perturbation_objectives = perturbation_objectives.pivot(
            on="control_name", values="perturbed_control_value"
        )

        if batch_objective_gradient is not None:
            objective_names = (
                batch_objective_gradient["objective_name"].unique().to_list()
            )
            batch_objective_gradient = batch_objective_gradient.pivot(
                on="objective_name",
                values=["objective_value", "total_objective_value"],
                separator=";",
            ).rename(
                {
                    **{f"objective_value;{name}": name for name in objective_names},
                    **{
                        f"total_objective_value;{name}": f"{name}.total"
                        for name in objective_names
                    },
                }
            )

        return {
            "batch_objective_gradient": batch_objective_gradient,
            "perturbation_objectives": perturbation_objectives,
            "batch_constraint_gradient": batch_constraint_gradient,
            "perturbation_constraints": perturbation_constraints,
        }

    def _on_batch_evaluation_finished(self, event: Event) -> None:
        logger.debug("Storing batch results dataframes")

        converted_results = tuple(
            convert_to_maximize(result) for result in event.results
        )
        results: list[FunctionResults | GradientResults] = []

        best_value = -np.inf
        best_results = None
        for item in converted_results:
            if isinstance(item, GradientResults):
                results.append(item)
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and item.functions.weighted_objective > best_value
            ):
                best_value = item.functions.weighted_objective
                best_results = item

        if best_results is not None:
            results = [best_results, *results]

        batch_dicts = {}
        for item in results:
            if item.batch_id not in batch_dicts:
                batch_dicts[item.batch_id] = {}

            if isinstance(item, FunctionResults):
                eval_results = self._store_function_results(item)
                batch_dicts[item.batch_id].update(eval_results)

            if isinstance(item, GradientResults):
                gradient_results = self._store_gradient_results(item)
                batch_dicts[item.batch_id].update(gradient_results)

        for batch_id, batch_dict in batch_dicts.items():
            self.data.batches.append(
                BatchStorageData(
                    batch_id=batch_id,
                    realization_controls=batch_dict.get("realization_controls"),
                    batch_objectives=batch_dict.get("batch_objectives"),
                    realization_objectives=batch_dict.get("realization_objectives"),
                    batch_constraints=batch_dict.get("batch_constraints"),
                    realization_constraints=batch_dict.get("realization_constraints"),
                    batch_objective_gradient=batch_dict.get("batch_objective_gradient"),
                    perturbation_objectives=batch_dict.get("perturbation_objectives"),
                    batch_constraint_gradient=batch_dict.get(
                        "batch_constraint_gradient"
                    ),
                    perturbation_constraints=batch_dict.get("perturbation_constraints"),
                )
            )

    def _on_optimization_finished(self, _) -> None:
        logger.debug("Storing final results Everest storage")

        merit_values = self._get_merit_values()
        if merit_values:
            # NOTE: Batch 0 is always an "accepted batch", and "accepted batches" are
            # batches with merit_flag , which means that it was an improvement
            self.data.batches[0].is_improvement = True

            improvement_batches = [
                b for b in self.data.batches if b.batch_objectives is not None
            ][1:]
            for i, b in enumerate(improvement_batches):
                merit_value = next(
                    (m.value for m in merit_values if (m.iter - 1) == i), None
                )
                if merit_value is None:
                    continue

                b.batch_objectives = b.batch_objectives.with_columns(
                    polars.lit(merit_value).alias("merit_value")
                )
                b.is_improvement = True
        else:
            max_total_objective = -np.inf
            for b in self.data.batches:
                if b.batch_objectives is not None:
                    total_objective = b.batch_objectives["total_objective_value"].item()
                    if total_objective > max_total_objective:
                        b.is_improvement = True
                        max_total_objective = total_objective

        self.write_to_output_dir()

    def get_optimal_result(self) -> OptimalResult | None:
        # Only used in tests, but re-created to ensure
        # same behavior as w/ old SEBA setup
        has_merit = any(
            "merit_value" in b.batch_objectives.columns
            for b in self.data.batches
            if b.batch_objectives is not None
        )

        def find_best_batch(
            filter_by, sort_by
        ) -> tuple[BatchStorageData | None, dict | None]:
            matching_batches = [b for b in self.data.batches if filter_by(b)]

            if not matching_batches:
                return None, None

            matching_batches.sort(key=sort_by)
            batch = matching_batches[0]
            controls_dict = batch.realization_controls.drop(
                [
                    "batch_id",
                    "simulation_id",
                    "realization",
                ]
            ).to_dicts()[0]

            return batch, controls_dict

        if has_merit:
            # Minimize merit
            batch, controls_dict = find_best_batch(
                filter_by=lambda b: (
                    b.batch_objectives is not None
                    and "merit_value" in b.batch_objectives.columns
                ),
                sort_by=lambda b: b.batch_objectives.select(
                    polars.col("merit_value").min()
                ).item(),
            )

            if batch is None:
                return None

            return OptimalResult(
                batch=batch.batch_id,
                controls=controls_dict,
                total_objective=batch.batch_objectives.select(
                    polars.col("total_objective_value").sample(n=1)
                ).item(),
            )
        else:
            # Maximize objective
            batch, controls_dict = find_best_batch(
                filter_by=lambda b: b.batch_objectives is not None
                and not b.batch_objectives.is_empty(),
                sort_by=lambda b: -b.batch_objectives.select(
                    polars.col("total_objective_value").sample(n=1)
                ).item(),
            )

            if batch is None:
                return None

            return OptimalResult(
                batch=batch.batch_id,
                controls=controls_dict,
                total_objective=batch.batch_objectives.select(
                    polars.col("total_objective_value")
                ).item(),
            )

    def _get_merit_values(self) -> list[_MeritValue]:
        # Read the file containing merit information.
        # The file should contain the following table header
        # Iter    F(x)    mu    alpha    Merit    feval    btracks    Penalty
        # :return: merit values indexed by the function evaluation number

        merit_file = Path(self._output_dir) / "dakota" / "OPT_DEFAULT.out"

        def _get_merit_fn_lines() -> list[str]:
            if os.path.isfile(merit_file):
                with open(merit_file, errors="replace", encoding="utf-8") as reader:
                    lines = reader.readlines()
                start_line_idx = -1
                for inx, line in enumerate(lines):
                    if "Merit" in line and "feval" in line:
                        start_line_idx = inx + 1
                    if start_line_idx > -1 and line.startswith("="):
                        return lines[start_line_idx:inx]
                if start_line_idx > -1:
                    return lines[start_line_idx:]
            return []

        def _parse_merit_line(merit_values_string: str) -> tuple[int, float] | None:
            values = []
            for merit_elem in merit_values_string.split():
                try:
                    values.append(float(merit_elem))
                except ValueError:
                    for elem in merit_elem.split("0x")[1:]:
                        values.append(float.fromhex("0x" + elem))
            if len(values) == 8:
                # Dakota starts counting at one, correct to be zero-based.
                return int(values[5]) - 1, values[4]
            return None

        merit_values = []
        if merit_file.exists():
            for line in _get_merit_fn_lines():
                value = _parse_merit_line(line)
                if value is not None:
                    merit_values.append(_MeritValue(iter=value[0], value=value[1]))

        return merit_values
