from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import polars as pl
from ropt.results import FunctionResults, GradientResults, Results

from ert.storage import LocalExperiment, open_storage
from ert.storage.local_ensemble import BatchDataframes
from ert.storage.local_experiment import _FunctionResults, _GradientResults
from everest.strings import EVEREST

logger = logging.getLogger(__name__)


class OptimizationDataframes(TypedDict, total=False):
    realization_weights: pl.DataFrame | None


class EverestStorage:
    @classmethod
    def from_storage_path(cls, storage_path: Path) -> LocalExperiment:
        """
        Creates everest storage from a storage path. Note: This
        requires there to be at least one initialized batch/ensemble
        for it to be possible to detect the experiment.
        """
        storage = open_storage(storage_path, mode="r")
        experiment = next(storage.experiments)
        assert isinstance(experiment, LocalExperiment)
        return experiment

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

    @classmethod
    def _ropt_to_df(
        cls,
        results: FunctionResults | GradientResults,
        field: str,
        *,
        values: list[str],
        select: list[str],
    ) -> pl.DataFrame:
        df = pl.from_pandas(
            results.to_dataframe(field, select=values).reset_index(),
        ).select(select + values)
        df = cls._rename_ropt_df_columns(df)
        df = cls._enforce_dtypes(df)

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

    @classmethod
    def _unpack_function_results(cls, results: FunctionResults) -> _FunctionResults:
        # We could select only objective values,
        # but we select all to also get the constraint values (if they exist)
        logger.debug("Storing function results")
        realization_objectives = cls._ropt_to_df(
            results,
            "evaluations",
            values=["objectives", "evaluation_info.sim_ids"],
            select=["batch_id", "realization", "objective"],
        )

        if results.functions is not None and results.functions.constraints is not None:
            realization_constraints = cls._ropt_to_df(
                results,
                "evaluations",
                values=["constraints", "evaluation_info.sim_ids"],
                select=["batch_id", "realization", "nonlinear_constraint"],
            )

            batch_constraints = cls._ropt_to_df(
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

        batch_objectives = cls._ropt_to_df(
            results,
            "functions",
            values=["objectives", "weighted_objective"],
            select=["batch_id", "objective"],
        )

        realization_controls = cls._ropt_to_df(
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
                batch_bound_constraint_violations = cls._ropt_to_df(
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
                batch_input_constraint_violations = cls._ropt_to_df(
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
                batch_output_constraint_violations = cls._ropt_to_df(
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

    @classmethod
    def _unpack_gradient_results(cls, results: GradientResults) -> _GradientResults:
        logger.debug("Storing gradient results")
        have_perturbed_constraints = (
            results.evaluations.perturbed_constraints is not None
        )
        perturbation_objectives = cls._ropt_to_df(
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
            batch_objective_gradient = cls._ropt_to_df(
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

    @classmethod
    def unpack_ropt_results(
        cls, optimizer_results: tuple[Results, ...]
    ) -> dict[int, BatchDataframes]:
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
                eval_results = cls._unpack_function_results(item)
                batch_dicts[item.batch_id].update(eval_results)

            if isinstance(item, GradientResults):
                gradient_results = cls._unpack_gradient_results(item)
                batch_dicts[item.batch_id].update(gradient_results)

        batch_dataframes: dict[int, BatchDataframes] = {}
        for batch_id, batch_dict in batch_dicts.items():
            batch_dataframes[batch_id] = {
                "realization_controls": batch_dict.get("realization_controls"),
                "batch_objectives": batch_dict.get("batch_objectives"),
                "realization_objectives": batch_dict.get("realization_objectives"),
                "batch_constraints": batch_dict.get("batch_constraints"),
                "realization_constraints": batch_dict.get("realization_constraints"),
                "batch_bound_constraint_violations": batch_dict.get(
                    "batch_bound_constraint_violations"
                ),
                "batch_input_constraint_violations": batch_dict.get(
                    "batch_input_constraint_violations"
                ),
                "batch_output_constraint_violations": batch_dict.get(
                    "batch_output_constraint_violations"
                ),
                "batch_objective_gradient": batch_dict.get("batch_objective_gradient"),
                "perturbation_objectives": batch_dict.get("perturbation_objectives"),
                "batch_constraint_gradient": batch_dict.get(
                    "batch_constraint_gradient"
                ),
                "perturbation_constraints": batch_dict.get("perturbation_constraints"),
            }

        return batch_dataframes
