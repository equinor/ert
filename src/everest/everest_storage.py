from __future__ import annotations

import json
import logging
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import numpy as np
import polars as pl
from ropt.results import FunctionResults, GradientResults, Results

from everest.config import EverestConfig
from everest.strings import EVEREST

logger = logging.getLogger(__name__)


@dataclass
class OptimalResult:
    batch: int
    controls: dict[str, Any]
    total_objective: float


def try_read_df(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


class BatchDataframes(TypedDict, total=False):
    realization_controls: pl.DataFrame | None
    batch_objectives: pl.DataFrame | None
    realization_objectives: pl.DataFrame | None
    batch_constraints: pl.DataFrame | None
    realization_constraints: pl.DataFrame | None
    batch_objective_gradient: pl.DataFrame | None
    perturbation_objectives: pl.DataFrame | None
    batch_constraint_gradient: pl.DataFrame | None
    perturbation_constraints: pl.DataFrame | None


class OptimizationDataframes(TypedDict, total=False):
    controls: pl.DataFrame | None
    objective_functions: pl.DataFrame | None
    nonlinear_constraints: pl.DataFrame | None
    realization_weights: pl.DataFrame | None


class BatchStorageData:
    BATCH_DATAFRAMES: ClassVar[list[str]] = [
        "realization_controls",
        "batch_objectives",
        "realization_objectives",
        "batch_constraints",
        "realization_constraints",
        "batch_objective_gradient",
        "perturbation_objectives",
        "batch_constraint_gradient",
        "perturbation_constraints",
    ]

    def __init__(self, path: Path):
        self._path = path

    @property
    def has_data(self) -> bool:
        return any(
            (self._path / f"{df_name}.parquet").exists()
            for df_name in self.BATCH_DATAFRAMES
        )

    @property
    def has_function_results(self) -> bool:
        return any(
            (self._path / f"{df_name}.parquet").exists()
            for df_name in _FunctionResults.__annotations__
        )

    @property
    def has_gradient_results(self) -> bool:
        return any(
            (self._path / f"{df_name}.parquet").exists()
            for df_name in _GradientResults.__annotations__
        )

    @staticmethod
    def _read_df_if_exists(path: Path) -> pl.DataFrame | None:
        if path.exists():
            return pl.read_parquet(path)
        return None

    @property
    def realization_controls(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "realization_controls.parquet")

    @property
    def batch_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "batch_objectives.parquet")

    @property
    def realization_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "realization_objectives.parquet")

    @property
    def batch_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "batch_constraints.parquet")

    @property
    def realization_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "realization_constraints.parquet")

    @property
    def batch_objective_gradient(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "batch_objective_gradient.parquet")

    @property
    def perturbation_objectives(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "perturbation_objectives.parquet")

    @property
    def batch_constraint_gradient(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "batch_constraint_gradient.parquet")

    @property
    def perturbation_constraints(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(self._path / "perturbation_constraints.parquet")

    def save_dataframes(self, dataframes: BatchDataframes) -> None:
        for df_name in self.BATCH_DATAFRAMES:
            df = dataframes.get(df_name)
            if isinstance(df, pl.DataFrame):
                df.write_parquet(self._path / f"{df_name}.parquet")

    @cached_property
    def is_improvement(self) -> bool:
        with open(self._path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)

        return bool(info["is_improvement"])

    @cached_property
    def batch_id(self) -> bool:
        with open(self._path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)

        return info["batch_id"]

    def write_metadata(self, is_improvement: bool) -> None:
        # Clear the cached prop for the new value to take place
        if "is_improvement" in self.__dict__:
            del self.is_improvement

        with open(self._path / "batch.json", encoding="utf-8") as f:
            info = json.load(f)
            info["is_improvement"] = is_improvement

        with open(self._path / "batch.json", "w", encoding="utf-8") as f:
            json.dump(
                info,
                f,
            )


class FunctionBatchStorageData(BatchStorageData):
    @property
    def realization_controls(self) -> pl.DataFrame:
        df = super().realization_controls
        assert df is not None
        return df

    @property
    def batch_objectives(self) -> pl.DataFrame:
        df = super().batch_objectives
        assert df is not None
        return df

    @property
    def realization_objectives(self) -> pl.DataFrame:
        df = super().realization_objectives
        assert df is not None
        return df


class GradientBatchStorageData(BatchStorageData):
    @property
    def perturbation_objectives(self) -> pl.DataFrame:
        df = super().perturbation_objectives
        assert df is not None
        return df


class OptimizationStorageData:
    EXPERIMENT_DATAFRAMES: ClassVar[list[str]] = [
        "controls",
        "objective_functions",
        "nonlinear_constraints",
        "realization_weights",
    ]

    def __init__(self, path: Path) -> None:
        self._path = path
        self.batches: list[BatchStorageData] = []

    @property
    def batches_with_function_results(self) -> list[FunctionBatchStorageData]:
        return [
            FunctionBatchStorageData(b._path)
            for b in self.batches
            if b.has_function_results
        ]

    @property
    def batches_with_gradient_results(self) -> list[GradientBatchStorageData]:
        return [
            GradientBatchStorageData(b._path)
            for b in self.batches
            if b.has_gradient_results
        ]

    @property
    def controls(self) -> pl.DataFrame | None:
        return pl.read_parquet(self._path / "controls.parquet")

    @property
    def objective_functions(self) -> pl.DataFrame | None:
        return pl.read_parquet(self._path / "objective_functions.parquet")

    @property
    def nonlinear_constraints(self) -> pl.DataFrame | None:
        return try_read_df(self._path / "nonlinear_constraints.parquet")

    @property
    def realization_weights(self) -> pl.DataFrame | None:
        return pl.read_parquet(self._path / "realization_weights.parquet")

    def save_dataframes(self, dataframes: OptimizationDataframes) -> None:
        for df_name in self.EXPERIMENT_DATAFRAMES:
            df = dataframes.get(df_name)
            if isinstance(df, pl.DataFrame):
                df.write_parquet(self._path / f"{df_name}.parquet")

    def simulation_to_geo_realization_map(self, batch_id: int) -> dict[int, int]:
        """
        Mapping from simulation ID to geo-realization
        """
        dummy_df = next(
            (
                b.realization_controls
                for b in self.batches_with_function_results
                if b.batch_id == batch_id
            ),
            None,
        )

        if dummy_df is None:
            return {}

        mapping = {}
        for d in dummy_df.select("realization", "simulation_id").to_dicts():
            mapping[int(d["simulation_id"])] = int(d["realization"])

        return mapping

    def read_from_experiment(self, experiment: _OptimizerOnlyExperiment) -> None:
        for ens in experiment.ensembles.values():
            self.batches.append(
                BatchStorageData(
                    path=ens.optimizer_mount_point,
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
        self._output_dir: Path = output_dir
        self._ensembles: dict[str, _OptimizerOnlyEnsemble] = {}

    @property
    def optimizer_mount_point(self) -> Path:
        if not (self._output_dir / "optimizer").exists():
            Path.mkdir(self._output_dir / "optimizer", parents=True)

        return self._output_dir / "optimizer"

    @property
    def ensembles(self) -> dict[str, _OptimizerOnlyEnsemble]:
        if not Path(self._output_dir / "ensembles").exists():
            return {}

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


class _FunctionResults(TypedDict):
    realization_controls: pl.DataFrame
    batch_objectives: pl.DataFrame
    realization_objectives: pl.DataFrame
    batch_constraints: pl.DataFrame | None
    realization_constraints: pl.DataFrame | None


class _GradientResults(TypedDict):
    batch_objective_gradient: pl.DataFrame | None
    perturbation_objectives: pl.DataFrame | None
    batch_constraint_gradient: pl.DataFrame | None
    perturbation_constraints: pl.DataFrame | None


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
        self._experiment = _OptimizerOnlyExperiment(self._output_dir)

        self.data = OptimizationStorageData(self._experiment.optimizer_mount_point)

    @staticmethod
    def _rename_ropt_df_columns(df: pl.DataFrame) -> pl.DataFrame:
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
            "evaluation_info.sim_ids": "simulation_id",
        }
        return df.rename({k: v for k, v in renames.items() if k in df.columns})

    @staticmethod
    def _enforce_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        dtypes = {
            "batch_id": pl.UInt32,
            "perturbation": pl.Int32,
            "realization": pl.UInt32,
            # -1 is used as a value in simulator cache.
            # thus we need signed, otherwise we could do unsigned
            "simulation_id": pl.Int32,
            "objective_name": pl.String,
            "control_name": pl.String,
            "constraint_name": pl.String,
            "total_objective_value": pl.Float64,
            "control_value": pl.Float64,
            "objective_value": pl.Float64,
            "constraint_value": pl.Float64,
            "perturbed_control_value": pl.Float64,
            "perturbed_objective_value": pl.Float64,
            "perturbed_constraint_value": pl.Float64,
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

    def _ropt_to_df(
        self,
        results: FunctionResults | GradientResults,
        field: str,
        *,
        values: list[str],
        select: list[str],
    ) -> pl.DataFrame:
        df = pl.from_pandas(
            results.to_dataframe(field, select=values).reset_index(),
        ).select(select + values)

        # The results from ropt do not contain any names, but indices referring
        # to control names, objective names, etc. The corresponding names can be
        # retrieved from the everest configuration and were stored in the init
        # method. Here we replace the indices with those names:
        ropt_to_everest_names = {
            "variable": self.data.controls["control_name"]
            if self.data.controls is not None
            else None,
            "objective": self.data.objective_functions["objective_name"]
            if self.data.objective_functions is not None
            else None,
            "nonlinear_constraint": (
                self.data.nonlinear_constraints["constraint_name"]
                if self.data.nonlinear_constraints is not None
                else None
            ),
            "realization": self.data.realization_weights["realization"]
            if self.data.realization_weights is not None
            else None,
        }
        df = df.with_columns(
            pl.col(ropt_name).replace_strict(dict(enumerate(everest_names)))  # type: ignore
            for ropt_name, everest_names in ropt_to_everest_names.items()
            if ropt_name in select
        )

        df = self._rename_ropt_df_columns(df)
        df = self._enforce_dtypes(df)

        return df

    @staticmethod
    def check_for_deprecated_seba_storage(config_file: str) -> None:
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

    def init(self, everest_config: EverestConfig) -> None:
        controls = pl.DataFrame(
            {
                "control_name": pl.Series(
                    everest_config.formatted_control_names, dtype=pl.String
                ),
            }
        )

        # TODO: The weight and normalization keys are only used by the everest api,
        # with everviz. They should be removed in the long run.
        weights = np.fromiter(
            (
                1.0 if obj.weight is None else obj.weight
                for obj in everest_config.objective_functions
            ),
            dtype=np.float64,
        )
        objective_functions = pl.DataFrame(
            {
                "objective_name": everest_config.objective_names,
                "weight": pl.Series(weights / sum(weights), dtype=pl.Float64),
                "scale": pl.Series(
                    [
                        1.0 if obj.scale is None else obj.scale
                        for obj in everest_config.objective_functions
                    ],
                    dtype=pl.Float64,
                ),
            }
        )

        nonlinear_constraints = (
            pl.DataFrame(
                {
                    "constraint_name": everest_config.constraint_names,
                }
            )
            if everest_config.output_constraints is not None
            else None
        )

        realization_weights = pl.DataFrame(
            {
                "realization": pl.Series(
                    everest_config.model.realizations, dtype=pl.UInt32
                ),
            }
        )

        self.data.save_dataframes(
            {
                "controls": controls,
                "objective_functions": objective_functions,
                "nonlinear_constraints": nonlinear_constraints,
                "realization_weights": realization_weights,
            }
        )

    def _store_function_results(self, results: FunctionResults) -> _FunctionResults:
        # We could select only objective values,
        # but we select all to also get the constraint values (if they exist)
        realization_objectives = self._ropt_to_df(
            results,
            "evaluations",
            values=["objectives", "evaluation_info.sim_ids"],
            select=["batch_id", "realization", "objective"],
        )

        if results.functions is not None and results.functions.constraints is not None:
            realization_constraints = self._ropt_to_df(
                results,
                "evaluations",
                values=["constraints", "evaluation_info.sim_ids"],
                select=["batch_id", "realization", "nonlinear_constraint"],
            )

            batch_constraints = self._ropt_to_df(
                results,
                "functions",
                values=["constraints"],
                select=["batch_id", "nonlinear_constraint"],
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

        batch_objectives = self._ropt_to_df(
            results,
            "functions",
            values=["objectives", "weighted_objective"],
            select=["batch_id", "objective"],
        )

        realization_controls = self._ropt_to_df(
            results,
            "evaluations",
            values=["variables", "evaluation_info.sim_ids"],
            select=["batch_id", "variable", "realization"],
        )

        realization_controls = realization_controls.pivot(
            on="control_name",
            values=["control_value"],
            separator=":",
        )

        batch_objectives = batch_objectives.pivot(
            on="objective_name",
            values=["objective_value"],
            separator=":",
        )

        realization_objectives = realization_objectives.pivot(
            on="objective_name",
            values="objective_value",
            index=[
                "batch_id",
                "realization",
                "simulation_id",
            ],
        )

        return {
            "realization_controls": realization_controls,
            "batch_objectives": batch_objectives,
            "realization_objectives": realization_objectives,
            "batch_constraints": batch_constraints,
            "realization_constraints": realization_constraints,
        }

    def _store_gradient_results(self, results: GradientResults) -> _GradientResults:
        have_perturbed_constraints = (
            results.evaluations.perturbed_constraints is not None
        )
        perturbation_objectives = self._ropt_to_df(
            results,
            "evaluations",
            values=(
                [
                    "variables",
                    "perturbed_variables",
                    "perturbed_objectives",
                    "evaluation_info.sim_ids",
                ]
                + (["perturbed_constraints"] if have_perturbed_constraints else [])
            ),
            select=(
                ["batch_id", "variable", "realization", "perturbation", "objective"]
                + (["nonlinear_constraint"] if have_perturbed_constraints else [])
            ),
        )

        if results.gradients is not None:
            have_constraints = results.gradients.constraints is not None
            batch_objective_gradient = self._ropt_to_df(
                results,
                "gradients",
                values=(
                    ["weighted_objective", "objectives"]
                    + (["constraints"] if have_constraints else [])
                ),
                select=(
                    ["batch_id", "variable", "objective"]
                    + (["nonlinear_constraint"] if have_constraints else [])
                ),
            )
        else:
            batch_objective_gradient = None

        if have_perturbed_constraints:
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
            "simulation_id", "control_value"
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

    def on_batch_evaluation_finished(
        self, optimizer_results: tuple[Results, ...]
    ) -> None:
        logger.debug("Storing batch results dataframes")

        results: list[FunctionResults | GradientResults] = []

        best_value = -np.inf
        best_results = None
        for item in optimizer_results:
            if isinstance(item, GradientResults):
                results.append(item)
            if (
                isinstance(item, FunctionResults)
                and item.functions is not None
                and item.functions.weighted_objective > best_value
            ):
                best_value = float(item.functions.weighted_objective)
                best_results = item

        if best_results is not None:
            results = [best_results, *results]

        batch_dicts: dict[int, Any] = {}
        for item in results:
            assert item.batch_id is not None

            if item.batch_id not in batch_dicts:
                batch_dicts[item.batch_id] = {}

            if isinstance(item, FunctionResults):
                eval_results = self._store_function_results(item)
                batch_dicts[item.batch_id].update(eval_results)

            if isinstance(item, GradientResults):
                gradient_results = self._store_gradient_results(item)
                batch_dicts[item.batch_id].update(gradient_results)

        for batch_id, batch_dict in batch_dicts.items():
            target_ensemble = self._experiment.get_ensemble_by_name(f"batch_{batch_id}")

            with open(
                target_ensemble.optimizer_mount_point / "batch.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "batch_id": batch_id,
                        "is_improvement": False,
                    },
                    f,
                )

            batch_data = BatchStorageData(path=target_ensemble.optimizer_mount_point)

            batch_data.save_dataframes(
                {
                    "realization_controls": batch_dict.get("realization_controls"),
                    "batch_objectives": batch_dict.get("batch_objectives"),
                    "realization_objectives": batch_dict.get("realization_objectives"),
                    "batch_constraints": batch_dict.get("batch_constraints"),
                    "realization_constraints": batch_dict.get(
                        "realization_constraints"
                    ),
                    "batch_objective_gradient": batch_dict.get(
                        "batch_objective_gradient"
                    ),
                    "perturbation_objectives": batch_dict.get(
                        "perturbation_objectives"
                    ),
                    "batch_constraint_gradient": batch_dict.get(
                        "batch_constraint_gradient"
                    ),
                    "perturbation_constraints": batch_dict.get(
                        "perturbation_constraints"
                    ),
                }
            )

            self.data.batches.append(batch_data)

    def on_optimization_finished(self) -> None:
        logger.debug("Storing final results Everest storage")

        merit_values = self._get_merit_values()
        if merit_values:
            # NOTE: Batch 0 is always an "accepted batch", and "accepted batches" are
            # batches with merit_flag , which means that it was an improvement
            self.data.batches[0].write_metadata(is_improvement=True)

            improvement_batches = self.data.batches_with_function_results[1:]
            for i, b in enumerate(improvement_batches):
                merit_value = next(
                    (m.value for m in merit_values if (m.iter - 1) == i), None
                )
                if merit_value is None:
                    continue

                b.save_dataframes(
                    {
                        "batch_objectives": b.batch_objectives.with_columns(
                            pl.lit(merit_value).alias("merit_value")
                        )
                    }
                )
                b.write_metadata(is_improvement=True)
        else:
            max_total_objective = -np.inf
            for b in self.data.batches_with_function_results:
                total_objective = b.batch_objectives["total_objective_value"].item()
                if total_objective > max_total_objective:
                    b.write_metadata(is_improvement=True)
                    max_total_objective = total_objective

    def get_optimal_result(self) -> OptimalResult | None:
        # Only used in tests, but re-created to ensure
        # same behavior as w/ old SEBA setup
        has_merit = any(
            "merit_value" in b.batch_objectives.columns
            for b in self.data.batches_with_function_results
        )

        def find_best_fn_eval_batch(
            filter_by: Callable[[FunctionBatchStorageData], bool],
            sort_by: Callable[[FunctionBatchStorageData], Any],
        ) -> tuple[FunctionBatchStorageData, dict[str, Any]] | None:
            matching_batches = [
                b for b in self.data.batches_with_function_results if filter_by(b)
            ]

            if not matching_batches:
                return None

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
            result = find_best_fn_eval_batch(
                filter_by=lambda b: "merit_value" in b.batch_objectives.columns,
                sort_by=lambda b: b.batch_objectives.select(
                    pl.col("merit_value").min()
                ).item(),
            )

            if result is None:
                return None

            batch, controls_dict = result

            return OptimalResult(
                batch=batch.batch_id,
                controls=controls_dict,
                total_objective=batch.batch_objectives.select(
                    pl.col("total_objective_value").sample(n=1)
                ).item(),
            )
        else:
            # Maximize objective
            result = find_best_fn_eval_batch(
                filter_by=lambda b: not b.batch_objectives.is_empty(),
                sort_by=lambda b: -b.batch_objectives.select(
                    pl.col("total_objective_value").sample(n=1)
                ).item(),
            )

            if result is None:
                return None

            batch, controls_dict = result

            return OptimalResult(
                batch=batch.batch_id,
                controls=controls_dict,
                total_objective=batch.batch_objectives.select(
                    pl.col("total_objective_value")
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
