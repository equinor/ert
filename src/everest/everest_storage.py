from __future__ import annotations

import json
import logging
import os
import traceback
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import numpy as np
import polars as pl
from ropt.results import FunctionResults, GradientResults, Results

from everest.config.objective_function_config import ObjectiveFunctionConfig
from everest.config.output_constraint_config import OutputConstraintConfig
from everest.strings import EVEREST

logger = logging.getLogger(__name__)


def try_read_df(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


class BatchDataframes(TypedDict, total=False):
    realization_controls: pl.DataFrame | None
    batch_objectives: pl.DataFrame | None
    realization_objectives: pl.DataFrame | None
    batch_constraints: pl.DataFrame | None
    realization_constraints: pl.DataFrame | None
    batch_bound_constraint_violations: pl.DataFrame | None
    batch_input_constraint_violations: pl.DataFrame | None
    batch_output_constraint_violations: pl.DataFrame | None
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
        "batch_bound_constraint_violations",
        "batch_input_constraint_violations",
        "batch_output_constraint_violations",
        "batch_objective_gradient",
        "perturbation_objectives",
        "batch_constraint_gradient",
        "perturbation_constraints",
    ]

    def __init__(self, path: Path) -> None:
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
    def batch_bound_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._path / "batch_bound_constraint_violations.parquet"
        )

    @property
    def batch_input_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._path / "batch_input_constraint_violations.parquet"
        )

    @property
    def batch_output_constraint_violations(self) -> pl.DataFrame | None:
        return self._read_df_if_exists(
            self._path / "batch_output_constraint_violations.parquet"
        )

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": self.realization_controls.drop(
                "batch_id", "realization", "simulation_id"
            ).to_dicts()[0],
            "objectives": self.batch_objectives.drop(
                "batch_id", "total_objective_value"
            ).to_dicts()[0],
            "total_objective_value": self.batch_objectives[
                "total_objective_value"
            ].item(),
            "realization_objectives": self.realization_objectives.drop(
                "batch_id"
            ).to_dicts(),
        }


class GradientBatchStorageData(BatchStorageData):
    @property
    def perturbation_objectives(self) -> pl.DataFrame:
        df = super().perturbation_objectives
        assert df is not None
        return df

    def to_dict(self) -> dict[str, Any]:
        objective_gradient = (
            self.batch_objective_gradient.drop("batch_id")
            .sort("control_name")
            .to_dicts()
            if self.batch_objective_gradient is not None
            else None
        )

        perturbation_objectives = (
            self.perturbation_objectives.drop("batch_id")
            .sort("realization", "perturbation")
            .to_dicts()
        )
        constraint_gradient_dicts = (
            self.batch_constraint_gradient.drop("batch_id")
            .sort("control_name")
            .to_dicts()
            if self.batch_constraint_gradient is not None
            else None
        )

        perturbation_gradient_dicts = (
            self.perturbation_constraints.drop("batch_id")
            .sort("realization", "perturbation")
            .to_dicts()
            if self.perturbation_constraints is not None
            else None
        )

        return {
            "objective_gradient_values": objective_gradient,
            "perturbation_objectives": perturbation_objectives,
            "constraint_gradient": constraint_gradient_dicts,
            "perturbation_constraints": perturbation_gradient_dicts,
        }


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
    batch_bound_constraint_violations: pl.DataFrame | None
    batch_input_constraint_violations: pl.DataFrame | None
    batch_output_constraint_violations: pl.DataFrame | None


class _GradientResults(TypedDict):
    batch_objective_gradient: pl.DataFrame | None
    perturbation_objectives: pl.DataFrame | None
    batch_constraint_gradient: pl.DataFrame | None
    perturbation_constraints: pl.DataFrame | None


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

    @property
    def is_empty(self) -> bool:
        return not any(b.has_data for b in self.data.batches)

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
            "bound_violation": "bound_constraint_violation",
            "linear_violation": "input_constraint_violation",
            "nonlinear_violation": "output_constraint_violation",
            "linear_constraint": "input_constraint_index",
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
            # -1 is used as a place-holder value.
            # thus we need signed, otherwise we could do unsigned
            "simulation_id": pl.Int32,
            "objective_name": pl.String,
            "control_name": pl.String,
            "constraint_name": pl.String,
            "input_constraint_index": pl.UInt32,
            "total_objective_value": pl.Float64,
            "control_value": pl.Float64,
            "objective_value": pl.Float64,
            "constraint_value": pl.Float64,
            "bound_constraint_violation": pl.Float64,
            "input_constraint_violation": pl.Float64,
            "output_constraint_violation": pl.Float64,
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
        df = self._rename_ropt_df_columns(df)
        df = self._enforce_dtypes(df)

        return df

    @staticmethod
    def check_for_deprecated_seba_storage(output_dir: str) -> None:
        if (Path(output_dir) / "seba.db").exists() or (
            Path(output_dir) / "seba.db.backup"
        ).exists():
            trace = "\n".join(traceback.format_stack())
            logging.getLogger(EVEREST).error(
                f"Tried opening old seba storage.Traceback: {trace}"
            )
            raise SystemExit(
                f"Trying to open old storage @ {output_dir}/seba.db."
                "This storage can only be opened with an "
                "ert[everest] version <= 12.1.2"
            )

    def read_from_output_dir(self) -> None:
        exp = _OptimizerOnlyExperiment(self._output_dir)
        self.data.read_from_experiment(exp)

    def init(
        self,
        formatted_control_names: list[str],
        objective_functions: list[ObjectiveFunctionConfig],
        output_constraints: list[OutputConstraintConfig] | None,
        realizations: list[int],
    ) -> None:
        controls = pl.DataFrame(
            {
                "control_name": pl.Series(formatted_control_names, dtype=pl.String),
            }
        )

        # TODO: The weight and normalization keys are only used by the everest api,
        # with everviz. They should be removed in the long run.
        weights = np.fromiter(
            (1.0 if obj.weight is None else obj.weight for obj in objective_functions),
            dtype=np.float64,
        )

        objective_functions_dataframe = pl.DataFrame(
            {
                "objective_name": [objective.name for objective in objective_functions],
                "weight": pl.Series(weights / sum(weights), dtype=pl.Float64),
                "scale": pl.Series(
                    [
                        1.0 if obj.scale is None else obj.scale
                        for obj in objective_functions
                    ],
                    dtype=pl.Float64,
                ),
            }
        )

        nonlinear_constraints = (
            pl.DataFrame(
                {
                    "constraint_name": [
                        constraint.name for constraint in output_constraints
                    ],
                }
            )
            if output_constraints
            else None
        )

        realization_weights = pl.DataFrame(
            {
                "realization": pl.Series(realizations, dtype=pl.UInt32),
            }
        )

        self.data.save_dataframes(
            {
                "controls": controls,
                "objective_functions": objective_functions_dataframe,
                "nonlinear_constraints": nonlinear_constraints,
                "realization_weights": realization_weights,
            }
        )

    def _store_function_results(self, results: FunctionResults) -> _FunctionResults:
        # We could select only objective values,
        # but we select all to also get the constraint values (if they exist)
        logger.debug("Storing function results")
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

        batch_bound_constraint_violations = None
        batch_input_constraint_violations = None
        batch_output_constraint_violations = None
        if results.constraint_info is not None:
            if results.constraint_info.bound_violation is not None:
                batch_bound_constraint_violations = self._ropt_to_df(
                    results,
                    "constraint_info",
                    values=["bound_violation"],
                    select=["batch_id", "variable"],
                )
                batch_bound_constraint_violations = (
                    batch_bound_constraint_violations.pivot(
                        on="control_name",
                        values=["bound_constraint_violation"],
                        separator=":",
                    )
                )
            if results.constraint_info.linear_violation is not None:
                batch_input_constraint_violations = self._ropt_to_df(
                    results,
                    "constraint_info",
                    values=["linear_violation"],
                    select=["batch_id", "linear_constraint"],
                )
                batch_input_constraint_violations = (
                    batch_input_constraint_violations.pivot(
                        on="input_constraint_index",
                        values=["input_constraint_violation"],
                        separator=":",
                    )
                )
            if results.constraint_info.nonlinear_violation is not None:
                batch_output_constraint_violations = self._ropt_to_df(
                    results,
                    "constraint_info",
                    values=["nonlinear_violation"],
                    select=["batch_id", "nonlinear_constraint"],
                )
                batch_output_constraint_violations = (
                    batch_output_constraint_violations.pivot(
                        on="constraint_name",
                        values=["output_constraint_violation"],
                        separator=":",
                    )
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
            "batch_bound_constraint_violations": batch_bound_constraint_violations,
            "batch_input_constraint_violations": batch_input_constraint_violations,
            "batch_output_constraint_violations": batch_output_constraint_violations,
        }

    def _store_gradient_results(self, results: GradientResults) -> _GradientResults:
        logger.debug("Storing gradient results")
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
                    "batch_bound_constraint_violations": batch_dict.get(
                        "batch_bound_constraint_violations"
                    ),
                    "batch_input_constraint_violations": batch_dict.get(
                        "batch_input_constraint_violations"
                    ),
                    "batch_output_constraint_violations": batch_dict.get(
                        "batch_output_constraint_violations"
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

        # This a somewhat arbitrary threshold, this should be a user choice
        # during visualization:
        CONSTRAINT_TOL = 1e-6

        max_total_objective = -np.inf
        for b in self.data.batches_with_function_results:
            total_objective = b.batch_objectives["total_objective_value"].item()
            bound_constraint_violation = (
                0.0
                if b.batch_bound_constraint_violations is None
                else (
                    b.batch_bound_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            input_constraint_violation = (
                0.0
                if b.batch_input_constraint_violations is None
                else (
                    b.batch_input_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            output_constraint_violation = (
                0.0
                if b.batch_output_constraint_violations is None
                else (
                    b.batch_output_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            if (
                max(
                    bound_constraint_violation,
                    input_constraint_violation,
                    output_constraint_violation,
                )
                < CONSTRAINT_TOL
                and total_objective > max_total_objective
            ):
                b.write_metadata(is_improvement=True)
                max_total_objective = total_objective

    def export_dataframes(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        batch_dfs_to_join = {}  # type: ignore
        realization_dfs_to_join = {}  # type: ignore
        perturbation_dfs_to_join = {}  # type: ignore

        batch_ids = [b.batch_id for b in self.data.batches]
        all_controls = (
            self.data.controls["control_name"].to_list()
            if self.data.controls is not None
            else []
        )

        def _try_append_df(
            batch_id: int,
            df: pl.DataFrame | None,
            target: dict[str, list[pl.DataFrame]],
        ) -> None:
            if df is not None:
                if batch_id not in target:  # type: ignore
                    target[batch.batch_id] = []  # type: ignore
                target[batch_id].append(df)  # type: ignore

        def try_append_batch_dfs(batch_id: int, *dfs: pl.DataFrame | None) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, batch_dfs_to_join)

        def try_append_realization_dfs(
            batch_id: int, *dfs: pl.DataFrame | None
        ) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, realization_dfs_to_join)

        def try_append_perturbation_dfs(
            batch_id: int, *dfs: pl.DataFrame | None
        ) -> None:
            for df_ in dfs:
                _try_append_df(batch_id, df_, perturbation_dfs_to_join)

        def pivot_gradient(df: pl.DataFrame) -> pl.DataFrame:
            pivoted_ = df.pivot(on="control_name", index="batch_id", separator=" wrt ")
            return pivoted_.rename(
                {
                    col: f"grad({col})"
                    for col in pivoted_.columns
                    if col != "batch_id" and col not in all_controls
                }
            )

        for batch in self.data.batches:
            if not batch.has_data:
                continue

            try_append_perturbation_dfs(
                batch.batch_id,
                batch.perturbation_objectives,
                batch.perturbation_constraints,
            )

            try_append_realization_dfs(
                batch.batch_id,
                batch.realization_objectives,
                batch.realization_controls,
                batch.realization_constraints,
            )

            if batch.batch_objective_gradient is not None:
                try_append_batch_dfs(
                    batch.batch_id, pivot_gradient(batch.batch_objective_gradient)
                )

            if batch.batch_constraint_gradient is not None:
                try_append_batch_dfs(
                    batch.batch_id,
                    pivot_gradient(batch.batch_constraint_gradient),
                )

            try_append_batch_dfs(
                batch.batch_id,
                batch.batch_objectives,
                batch.batch_constraints,
            )

        def _join_by_batch(
            dfs: dict[int, list[pl.DataFrame]], on: list[str]
        ) -> list[pl.DataFrame]:
            """
            Creates one dataframe per batch, with one column per input/output,
            including control, objective, constraint, gradient value.
            """
            dfs_to_concat_ = []
            for batch_id in batch_ids:
                if batch_id not in dfs:
                    continue

                batch_df_ = dfs[batch_id][0]
                for bdf_ in dfs[batch_id][1:]:
                    if set(all_controls).issubset(set(bdf_.columns)) and set(
                        all_controls
                    ).issubset(set(batch_df_.columns)):
                        bdf_ = bdf_.drop(all_controls)

                    batch_df_ = batch_df_.join(
                        bdf_,
                        on=on,
                    )

                dfs_to_concat_.append(batch_df_)

            return dfs_to_concat_

        batch_dfs_to_concat = _join_by_batch(batch_dfs_to_join, on=["batch_id"])
        batch_df = pl.concat(batch_dfs_to_concat, how="diagonal")

        realization_dfs_to_concat = _join_by_batch(
            realization_dfs_to_join, on=["batch_id", "realization", "simulation_id"]
        )
        realization_df = pl.concat(realization_dfs_to_concat, how="diagonal")

        perturbation_dfs_to_concat = _join_by_batch(
            perturbation_dfs_to_join, on=["batch_id", "realization", "perturbation"]
        )
        if perturbation_dfs_to_concat:
            # Perturbations exists, proceed as normal
            perturbation_df = pl.concat(perturbation_dfs_to_concat, how="diagonal")
            pert_real_df = pl.concat([realization_df, perturbation_df], how="diagonal")
        else:
            # Discrete methods never have perturbations,
            # append an empty (i.e., null) column
            pert_real_df = realization_df.with_columns(
                pl.lit(None).alias("perturbation")
            )

        pert_real_df = pert_real_df.select(
            "batch_id",
            "realization",
            "perturbation",
            *list(
                set(pert_real_df.columns) - {"batch_id", "realization", "perturbation"}
            ),
        )

        # Avoid name collisions when joining with simulations
        batch_df_renamed = batch_df.rename(
            {
                col: f"batch_{col}"
                for col in batch_df.columns
                if col != "batch_id" and not col.startswith("grad")
            }
        )
        combined_df = pert_real_df.join(
            batch_df_renamed, on="batch_id", how="full", coalesce=True
        )

        def _sort_df(df: pl.DataFrame, index: list[str]) -> pl.DataFrame:
            sorted_cols = index + sorted(set(df.columns) - set(index))
            df_ = df.select(sorted_cols).sort(by=index)
            return df_

        return (
            _sort_df(
                combined_df,
                ["batch_id", "realization", "simulation_id", "perturbation"],
            ),
            _sort_df(
                pert_real_df,
                [
                    "batch_id",
                    "realization",
                    "perturbation",
                    "simulation_id",
                ],
            ),
            _sort_df(batch_df, ["batch_id", "total_objective_value"]),
        )

    def export_everest_opt_results_to_csv(self) -> Path:
        batches_with_data = ",".join(
            {str(b.batch_id) for b in self.data.batches if b.has_data}
        )
        full_path = (
            self._output_dir
            / f"experiment_results_batches::{','.join(batches_with_data)}.csv"
        )

        # Find old csv to delete
        existing_csv = next(
            (
                Path(f)
                for f in os.listdir(self._output_dir)
                if f.startswith("experiment_results_batches::")
            ),
            None,
        )

        if (
            existing_csv is not None
            and existing_csv.exists()
            and (self._output_dir / existing_csv) != full_path
        ):
            # New batches are added -> overwrite existing csv
            os.remove(existing_csv)

        if not os.path.exists(full_path):
            combined_df, _, _ = self.export_dataframes()
            combined_df.write_csv(full_path)
        return full_path
