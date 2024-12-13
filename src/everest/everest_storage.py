from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import polars
from numpy.core.numeric import Infinity
from ropt.enums import EventType
from ropt.plan import BasicOptimizer, Event
from ropt.results import FunctionResults, GradientResults, convert_to_maximize
from seba_sqlite import sqlite_storage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class OptimalResult:
    batch: int
    controls: list[Any]
    total_objective: float

    @staticmethod
    def from_seba_optimal_result(
        o: sqlite_storage.OptimalResult | None = None,
    ) -> OptimalResult | None:
        if o is None:
            return None

        # Note: ROPT results are 1-indexed now, and seba keeps its own counter
        # +1'ing here corrects that discrepancy.
        return OptimalResult(
            batch=o.batch + 1, controls=o.controls, total_objective=o.total_objective
        )


def try_read_df(path: Path) -> polars.DataFrame | None:
    return polars.read_parquet(path) if path.exists() else None


@dataclass
class BatchDataFrames:
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
        dataframes = {}

        if self.batch_objectives is not None:
            dataframes["batch_objectives"] = self.batch_objectives

        if self.realization_objectives is not None:
            dataframes["realization_objectives"] = self.realization_objectives

        if self.realization_controls is not None:
            dataframes["realization_controls"] = self.realization_controls

        if self.batch_constraints is not None:
            dataframes["batch_constraints"] = self.batch_constraints

        if self.realization_constraints is not None:
            dataframes["realization_constraints"] = self.realization_constraints

        if self.batch_objective_gradient is not None:
            dataframes["batch_objective_gradient"] = self.batch_objective_gradient

        if self.perturbation_objectives is not None:
            dataframes["perturbation_objectives"] = self.perturbation_objectives

        if self.batch_constraint_gradient is not None:
            dataframes["batch_constraint_gradient"] = self.batch_constraint_gradient

        if self.perturbation_constraints is not None:
            dataframes["perturbation_constraints"] = self.perturbation_constraints

        return dataframes


@dataclass
class EverestStorageDataFrames:
    batches: list[BatchDataFrames] = field(default_factory=list)
    time_ended: datetime.date | None = None
    initial_values: polars.DataFrame | None = None
    objective_functions: polars.DataFrame | None = None
    nonlinear_constraints: polars.DataFrame | None = None
    realization_weights: polars.DataFrame | None = None

    def write_to_experiment(
        self, experiment: _OptimizerOnlyExperiment, write_csv=False
    ):
        # Stored in ensembles instead
        # self.batch_objectives.write_parquet(path / "objective_data.parquet")
        # self.gradient_evaluation.write_parquet(path / "gradient_evaluation.parquet")
        # self.gradient.write_parquet(path / "gradient.parquet")

        # The stuff under experiment should maybe be JSON?
        # 2DO PICK ONE, for now doing both for proof of concept

        with open(
            experiment.optimizer_mount_point / "initial_values.json",
            mode="w+",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(self.initial_values.to_dicts()))

        self.initial_values.write_parquet(
            experiment.optimizer_mount_point / "initial_values.parquet"
        )

        with open(
            experiment.optimizer_mount_point / "objective_functions.json",
            mode="w+",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(self.objective_functions.to_dicts()))

        self.objective_functions.write_parquet(
            experiment.optimizer_mount_point / "objective_functions.parquet"
        )

        if self.nonlinear_constraints is not None:
            with open(
                experiment.optimizer_mount_point / "nonlinear_constraints.json",
                mode="w+",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(self.nonlinear_constraints.to_dicts()))
            self.nonlinear_constraints.write_parquet(
                experiment.optimizer_mount_point / "nonlinear_constraints.parquet"
            )

        if self.realization_weights is not None:
            with open(
                experiment.optimizer_mount_point / "realization_weights.json",
                mode="w+",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(self.realization_weights.to_dicts()))

            self.realization_weights.write_parquet(
                experiment.optimizer_mount_point / "realization_weights.parquet"
            )

        for batch_data in self.batches:
            ensemble = experiment.get_ensemble_by_name(f"batch_{batch_data.batch_id}")
            with open(
                ensemble.optimizer_mount_point / "batch.json", "w+", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "id": batch_data.batch_id,
                        "is_improvement": batch_data.is_improvement,
                    },
                    f,
                )

            for df_key, df in batch_data.existing_dataframes.items():
                df.write_parquet(ensemble.optimizer_mount_point / f"{df_key}.parquet")

        if write_csv:
            self.initial_values.write_csv(
                experiment.optimizer_mount_point / "initial_values.csv"
            )

            self.objective_functions.write_csv(
                experiment.optimizer_mount_point / "objective_functions.csv"
            )

            if self.nonlinear_constraints is not None:
                self.nonlinear_constraints.write_csv(
                    experiment.optimizer_mount_point / "nonlinear_constraints.csv"
                )

            if self.realization_weights is not None:
                self.realization_weights.write_csv(
                    experiment.optimizer_mount_point / "realization_weights.csv"
                )

            for batch_data in self.batches:
                ensemble = experiment.get_ensemble_by_name(
                    f"batch_{batch_data.batch_id}"
                )
                for df_key, df in batch_data.existing_dataframes.items():
                    df.write_csv(ensemble.optimizer_mount_point / f"{df_key}.csv")
                    df.write_json(ensemble.optimizer_mount_point / f"{df_key}.json")

    def read_from_experiment(self, experiment: _OptimizerOnlyExperiment) -> None:
        self.initial_values = polars.read_parquet(
            experiment.optimizer_mount_point / "initial_values.parquet"
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

        for name, ens in experiment.ensembles.items():
            batch_id = int(name.split("_")[1])

            batch_objectives = try_read_df(
                ens.optimizer_mount_point / "batch_objectives.parquet"
            )
            realization_objectives = try_read_df(
                ens.optimizer_mount_point / "realization_objectives.parquet"
            )
            batch_constraints = try_read_df(
                ens.optimizer_mount_point / "batch_constraints.parquet"
            )
            realization_constraints = try_read_df(
                ens.optimizer_mount_point / "realization_constraints.parquet"
            )
            batch_objective_gradient = try_read_df(
                ens.optimizer_mount_point / "batch_objective_gradient.parquet"
            )
            perturbation_objectives = try_read_df(
                ens.optimizer_mount_point / "perturbation_objectives.parquet"
            )
            batch_constraint_gradient = try_read_df(
                ens.optimizer_mount_point / "batch_constraint_gradient.parquet"
            )
            perturbation_constraints = try_read_df(
                ens.optimizer_mount_point / "perturbation_constraints.parquet"
            )

            realization_controls = try_read_df(
                ens.optimizer_mount_point / "realization_controls.parquet"
            )

            with open(ens.optimizer_mount_point / "batch.json", encoding="utf-8") as f:
                info = json.load(f)
                batch_id = info["id"]
                is_improvement = info["is_improvement"]

            self.batches.append(
                BatchDataFrames(
                    batch_id,
                    realization_controls,
                    batch_objectives,
                    realization_objectives,
                    batch_constraints,
                    realization_constraints,
                    batch_objective_gradient,
                    perturbation_objectives,
                    batch_constraint_gradient,
                    perturbation_constraints,
                    is_improvement,
                )
            )

        self.batches.sort(key=lambda b: b.batch_id)


class _OptimizerOnlyEnsemble:
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    @property
    def optimizer_mount_point(self) -> Path:
        if not (self._output_dir / "optimizer").exists():
            Path.mkdir(self._output_dir / "optimizer", parents=True)

        return self._output_dir / "optimizer"


class _OptimizerOnlyExperiment:
    def __init__(self, output_dir: Path):
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
class _EvaluationResults:
    realization_controls: polars.DataFrame
    batch_objectives: polars.DataFrame
    realization_objectives: polars.DataFrame
    batch_constraints: polars.DataFrame | None
    realization_constraints: polars.DataFrame | None


@dataclass
class _GradientResults:
    batch_objective_gradient: polars.DataFrame
    perturbation_objectives: polars.DataFrame
    batch_constraint_gradient: polars.DataFrame | None
    perturbation_constraints: polars.DataFrame | None


class EverestStorage:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        self._initialized = False
        self._control_ensemble_id = 0
        self._gradient_ensemble_id = 0

        self._output_dir = output_dir
        self._merit_file: Path | None = None
        self.data = EverestStorageDataFrames()

    def write_to_output_dir(self) -> None:
        exp = _OptimizerOnlyExperiment(self._output_dir)

        # csv writing mostly for dev/debugging/quick inspection
        self.data.write_to_experiment(exp, write_csv=True)

    def read_from_output_dir(self) -> None:
        exp = _OptimizerOnlyExperiment(self._output_dir)
        self.data.read_from_experiment(exp)
        self._initialized = True

    def observe_optimizer(
        self,
        optimizer: BasicOptimizer,
        merit_file: Path,
    ) -> None:
        # We only need this file if we are observing a running ROPT instance
        # (using dakota backend)
        self._merit_file = merit_file

        # Q: Do these observers have to be explicitly disconnected/destroyed?
        optimizer.add_observer(EventType.START_OPTIMIZER_STEP, self._initialize)
        optimizer.add_observer(
            EventType.FINISHED_EVALUATION, self._handle_finished_batch_event
        )
        optimizer.add_observer(
            EventType.FINISHED_OPTIMIZER_STEP,
            self._handle_finished_event,
        )

    @property
    def experiment(self) -> _OptimizerOnlyExperiment:
        # Should be replaced with ERT experiment
        # in the long run
        return self._experiment

    @staticmethod
    def _convert_names(control_names):
        converted_names = []
        for name in control_names:
            converted = f"{name[0]}_{name[1]}"
            if len(name) > 2:
                converted += f"-{name[2]}"
            converted_names.append(converted)
        return converted_names

    @property
    def file(self):
        return self._database.location

    def _initialize(self, event):
        if self._initialized:
            return
        self._initialized = True

        config = event.config
        self.data.initial_values = polars.DataFrame(
            {
                "control_name": polars.Series(
                    self._convert_names(config.variables.names), dtype=polars.String
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
                "normalization": polars.Series(
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
                    config.realizations.names, dtype=polars.UInt16
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
                select=["objectives", "constraints", "evaluation_ids"],
            ).reset_index(),
        ).drop("plan_id")
        batch_objectives = polars.from_pandas(
            results.to_dataframe(
                "functions",
                select=["objectives", "weighted_objective"],
            ).reset_index()
        ).drop("plan_id")

        realization_controls = polars.from_pandas(
            results.to_dataframe(
                "evaluations", select=["variables", "evaluation_ids"]
            ).reset_index()
        ).drop("plan_id")

        realization_controls = self._rename_columns(realization_controls)
        realization_controls = self._enforce_dtypes(realization_controls)

        realization_controls = realization_controls.pivot(
            on="control_name",
            values=["control_value"],  # , "scaled_control_value"]
            separator=":",
        )

        try:
            batch_constraints = polars.from_pandas(
                results.to_dataframe("nonlinear_constraints").reset_index()
            ).drop("plan_id")
        except AttributeError:
            batch_constraints = None

        realization_constraints = None

        batch_objectives = self._rename_columns(batch_objectives)
        batch_objectives = self._enforce_dtypes(batch_objectives)

        realization_objectives = self._rename_columns(realization_objectives)
        realization_objectives = self._enforce_dtypes(realization_objectives)

        batch_objectives = batch_objectives.pivot(
            on="objective_name",
            values=["objective_value"],
            separator=":",
        )

        if batch_constraints is not None:
            batch_constraints = batch_constraints.rename(
                {
                    "nonlinear_constraint": "constraint_name",
                    "values": "constraint_value",
                    "violations": "constraint_violation",
                }
            )

            constraint_names = batch_constraints["constraint_name"].unique().to_list()

            batch_constraints = batch_constraints.pivot(
                on="constraint_name",
                values=[
                    "constraint_value",
                    "constraint_violation",
                ],
                separator=";",
            ).rename(
                {
                    **{f"constraint_value;{name}": name for name in constraint_names},
                    **{
                        f"constraint_violation;{name}": f"{name}.violation"
                        for name in constraint_names
                    },
                }
            )

            # remove from main table, and create separate constraints table
            realization_constraints = realization_objectives[
                "result_id",
                "batch_id",
                "realization",
                "simulation_id",
                "constraint_name",
                "constraint_value",
            ]
            realization_constraints = realization_constraints.pivot(
                values=["constraint_value"], on="constraint_name"
            )
            realization_objectives = realization_objectives.drop(
                [c for c in realization_objectives.columns if "constraint" in c.lower()]
            )
            batch_objectives = batch_objectives.drop(
                [c for c in batch_objectives.columns if "constraint" in c.lower()]
            )

        realization_objectives = realization_objectives.pivot(
            values="objective_value",
            index=[
                "result_id",
                "batch_id",
                "realization",
                "simulation_id",
            ],
            columns="objective_name",
        )

        return _EvaluationResults(
            realization_controls,
            batch_objectives,
            realization_objectives,
            batch_constraints,
            realization_constraints,
        )

    @staticmethod
    def _rename_columns(df: polars.DataFrame):
        scaled_cols = [c for c in df.columns if c.lower().startswith("scaled")]
        if len(scaled_cols) > 0:
            raise ValueError("Don't store scaled columns")

        renames = {
            "objective": "objective_name",
            "weighted_objective": "total_objective_value",
            "variable": "control_name",
            "variables": "control_value",
            "objectives": "objective_value",
            "constraints": "constraint_value",
            "nonlinear_constraint": "constraint_name",
            "scaled_constraints": "scaled_constraint_value",
            "scaled_objectives": "scaled_objective_value",
            "perturbed_variables": "perturbed_control_value",
            "perturbed_objectives": "perturbed_objective_value",
            "perturbed_constraints": "perturbed_constraint_value",
            "scaled_perturbed_objectives": "scaled_perturbed_objective_value",
            "scaled_perturbed_constraints": "scaled_perturbed_constraint_value",
            "scaled_variables": "scaled_control_value",
            "evaluation_ids": "simulation_id",
        }
        return df.rename({k: v for k, v in renames.items() if k in df.columns})

    @staticmethod
    def _enforce_dtypes(df: polars.DataFrame):
        dtypes = {
            "batch_id": polars.UInt16,
            "result_id": polars.UInt16,
            "perturbation": polars.UInt16,
            "realization": polars.UInt16,
            "simulation_id": polars.UInt16,
            "objective_name": polars.String,
            "control_name": polars.String,
            "constraint_name": polars.String,
            "total_objective_value": polars.Float64,
            "control_value": polars.Float64,
            "objective_value": polars.Float64,
            "constraint_value": polars.Float64,
            "scaled_constraint_value": polars.Float64,
            "scaled_objective_value": polars.Float64,
            "perturbed_control_value": polars.Float64,
            "perturbed_objective_value": polars.Float64,
            "perturbed_constraint_value": polars.Float64,
            "scaled_perturbed_objective_value": polars.Float64,
            "scaled_perturbed_constraint_value": polars.Float64,
            "scaled_control_value": polars.Float64,
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

    def _store_gradient_results(self, results: FunctionResults) -> _GradientResults:
        perturbation_objectives = polars.from_pandas(
            results.to_dataframe("evaluations").reset_index()
        ).drop("plan_id")
        perturbation_objectives = perturbation_objectives.drop(
            c for c in perturbation_objectives.columns if c.lower().startswith("scaled")
        )

        try:
            # ROPT_NOTE: Why is this sometimes None? How can we know if it is
            # expected to be None?
            batch_objective_gradient = polars.from_pandas(
                results.to_dataframe("gradients").reset_index()
            ).drop("plan_id")
        except AttributeError:
            batch_objective_gradient = None

        if batch_objective_gradient is not None:
            batch_objective_gradient = batch_objective_gradient.drop(
                c
                for c in batch_objective_gradient.columns
                if c.lower().startswith("scaled")
            )
            batch_objective_gradient = self._rename_columns(batch_objective_gradient)
            batch_objective_gradient = self._enforce_dtypes(batch_objective_gradient)

        perturbation_objectives = self._rename_columns(perturbation_objectives)
        perturbation_objectives = self._rename_columns(perturbation_objectives)

        if "constraint_name" in perturbation_objectives:
            perturbation_constraints = (
                perturbation_objectives[
                    "result_id",
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
                # ROPT_NOTE: Will this ever happen? We get constraints
                # but no "gradient" field in the results.
                batch_constraint_gradient = batch_objective_gradient[
                    "result_id",
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

        # All that remains in perturbation_objectives is
        # control values per realization, which is redundant to store here.

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

        return _GradientResults(
            batch_objective_gradient,
            perturbation_objectives,
            batch_constraint_gradient,
            perturbation_constraints,
        )

    def _handle_finished_batch_event(self, event: Event):
        logger.debug("Storing batch results dataframes")

        converted_results = tuple(
            convert_to_maximize(result) for result in event.results
        )
        results: list[FunctionResults | GradientResults] = []

        # Q: Maybe this whole clause can be removed?
        # Not sure why it is there, putting the best function result first
        # +-----------------------------------------------------------------+
        # |                                                                 |
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
        # |                                                                 |
        # +-----------------------------------------------------------------+
        last_batch = -1

        batches = {}
        for item in results:
            if item.batch_id not in batches:
                batches[item.batch_id] = {}

            if isinstance(item, FunctionResults):
                eval_results = self._store_function_results(item)

                batches[item.batch_id]["realization_controls"] = (
                    eval_results.realization_controls
                )
                batches[item.batch_id]["batch_objectives"] = (
                    eval_results.batch_objectives
                )
                batches[item.batch_id]["realization_objectives"] = (
                    eval_results.realization_objectives
                )
                batches[item.batch_id]["batch_constraints"] = (
                    eval_results.batch_constraints
                )
                batches[item.batch_id]["realization_constraints"] = (
                    eval_results.realization_constraints
                )

            if isinstance(item, GradientResults):
                gradient_results = self._store_gradient_results(item)

                batches[item.batch_id]["batch_objective_gradient"] = (
                    gradient_results.batch_objective_gradient
                )
                batches[item.batch_id]["perturbation_objectives"] = (
                    gradient_results.perturbation_objectives
                )
                batches[item.batch_id]["batch_constraint_gradient"] = (
                    gradient_results.batch_constraint_gradient
                )
                batches[item.batch_id]["perturbation_constraints"] = (
                    gradient_results.perturbation_constraints
                )

            if item.batch_id != last_batch:
                pass
                # Q: Could apply timestamps here but, necessary?..
                #    self._database.set_batch_ended
            last_batch = item.batch_id

        for batch_id, info in batches.items():
            self.data.batches.append(
                BatchDataFrames(
                    batch_id=batch_id,
                    realization_controls=info.get("realization_controls"),
                    batch_objectives=info.get("batch_objectives"),
                    realization_objectives=info.get("realization_objectives"),
                    batch_constraints=info.get("batch_constraints"),
                    realization_constraints=info.get("realization_constraints"),
                    batch_objective_gradient=info.get("batch_objective_gradient"),
                    perturbation_objectives=info.get("perturbation_objectives"),
                    batch_constraint_gradient=info.get("batch_constraint_gradient"),
                    perturbation_constraints=info.get("perturbation_constraints"),
                )
            )

    def _handle_finished_event(self, event):
        logger.debug("Storing final results in the sqlite database")

        merit_values = self._get_merit_values()
        if merit_values:
            # NOTE: Batch 0 is always an "accepted batch", and "accepted batches" are
            # batches with merit_flag , which means that it was an improvement
            # Should/could
            self.data.batches[0].is_improvement = True
            improvement_batches = [
                b for b in self.data.batches if b.batch_objectives is not None
            ][1:]
            for i, b in enumerate(improvement_batches):
                merit_value = next(
                    (m["value"] for m in merit_values if (m["iter"] - 1) == i), None
                )
                if merit_value is None:
                    continue

                b.batch_objectives = b.batch_objectives.with_columns(
                    polars.lit(merit_value).alias("merit_value")
                )
                b.is_improvement = True
        else:
            max_total_objective = -Infinity
            for b in self.data.batches:
                if b.batch_objectives is not None:
                    total_objective = b.batch_objectives["total_objective_value"].item()
                    if total_objective > max_total_objective:
                        b.is_improvement = True
                        max_total_objective = total_objective

        self.write_to_output_dir()

    def get_optimal_result(self) -> OptimalResult | None:
        # Only used in tests, not super important
        has_merit = any(
            "merit_value" in b.batch_objectives.columns
            for b in self.data.batches
            if b.batch_objectives is not None
        )

        def find_best_batch(filter_by, sort_by):
            matching_batches = [b for b in self.data.batches if filter_by(b)]

            if not matching_batches:
                return None

            matching_batches.sort(key=sort_by)
            batch = matching_batches[0]
            controls_dict = batch.realization_controls.drop(
                [
                    "result_id",
                    "batch_id",
                    "simulation_id",
                    "realization",
                    *[
                        c
                        for c in batch.realization_controls.columns
                        if c.endswith(".scaled")  # don't need scaled control values
                    ],
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

            return OptimalResult(
                batch=batch.batch_id,
                controls=controls_dict,
                total_objective=batch.batch_objectives.select(
                    polars.col("total_objective_value")
                ).item(),
            )

    @staticmethod
    def _get_merit_fn_lines(merit_path):
        if os.path.isfile(merit_path):
            with open(merit_path, errors="replace", encoding="utf-8") as reader:
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

    @staticmethod
    def _parse_merit_line(merit_values_string):
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

    def _get_merit_values(self):
        # Read the file containing merit information.
        # The file should contain the following table header
        # Iter    F(x)    mu    alpha    Merit    feval    btracks    Penalty
        # :return: merit values indexed by the function evaluation number
        # example:
        #     0: merit_value_0
        #     1: merit_value_1
        #     2  merit_value_2
        #     ...
        # ]
        merit_values = []
        if self._merit_file.exists():
            for line in EverestStorage._get_merit_fn_lines(self._merit_file):
                value = EverestStorage._parse_merit_line(line)
                if value is not None:
                    merit_values.append({"iter": value[0], "value": value[1]})
        return merit_values
