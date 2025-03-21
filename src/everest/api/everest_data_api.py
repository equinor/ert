import os
from pathlib import Path
from typing import Any

import polars as pl

from ert.storage import open_storage
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage


class EverestDataAPI:
    def __init__(self, config: EverestConfig, filter_out_gradient: bool = True):
        self._config = config
        output_folder = config.optimization_output_dir
        assert output_folder
        self._ever_storage = EverestStorage(Path(output_folder))

        if os.path.exists(output_folder):
            self._ever_storage.read_from_output_dir()

    @property
    def batches(self) -> list[int]:
        return sorted(
            b.batch_id for b in self._ever_storage.data.batches_with_function_results
        )

    @property
    def accepted_batches(self) -> list[int]:
        return sorted(
            b.batch_id for b in self._ever_storage.data.batches if b.is_improvement
        )

    @property
    def objective_function_names(self) -> list[str]:
        if self._ever_storage.data.objective_functions is None:
            return []
        return sorted(
            self._ever_storage.data.objective_functions["objective_name"]
            .unique()
            .to_list()
        )

    @property
    def output_constraint_names(self) -> list[str]:
        return (
            sorted(
                self._ever_storage.data.nonlinear_constraints["constraint_name"]
                .unique()
                .to_list()
            )
            if self._ever_storage.data.nonlinear_constraints is not None
            else []
        )

    @property
    def realizations(self) -> list[int]:
        if not self._ever_storage.data.batches_with_function_results:
            return []
        return sorted(
            self._ever_storage.data.batches_with_function_results[0]
            .realization_objectives["realization"]
            .unique()
            .to_list()
        )

    @property
    def simulations(self) -> list[int]:
        if not self._ever_storage.data.batches_with_function_results:
            return []
        return sorted(
            self._ever_storage.data.batches_with_function_results[0]
            .realization_objectives["simulation_id"]
            .unique()
            .to_list()
        )

    @property
    def control_names(self) -> list[str]:
        assert self._ever_storage.data.controls is not None
        return sorted(
            self._ever_storage.data.controls["control_name"].unique().to_list()
        )

    @property
    def control_values(self) -> list[dict[str, Any]]:
        all_control_names = (
            self._ever_storage.data.controls["control_name"].to_list()
            if self._ever_storage.data.controls is not None
            else []
        )
        new = []
        for batch in self._ever_storage.data.batches_with_function_results:
            for controls_dict in batch.realization_controls.to_dicts():
                for name in all_control_names:
                    new.append(
                        {
                            "control": name,
                            "batch": batch.batch_id,
                            "value": controls_dict[name],
                        }
                    )

        return new

    @property
    def objective_values(self) -> list[dict[str, Any]]:
        obj_values = []
        for b in self._ever_storage.data.batches_with_function_results:
            for (
                geo_realization,
                simulation_id,
            ), df in b.realization_objectives.sort(
                ["realization", "simulation_id"]
            ).group_by(["realization", "simulation_id"], maintain_order=True):
                for obj_dict in (
                    self._ever_storage.data.objective_functions.sort(
                        ["objective_name"]
                    ).to_dicts()
                    if self._ever_storage.data.objective_functions is not None
                    else []
                ):
                    obj_name = obj_dict["objective_name"]
                    obj_values.append(
                        {
                            "batch": int(b.batch_id),
                            "realization": int(geo_realization),  # type: ignore
                            "simulation": int(simulation_id),  # type: ignore
                            "function": obj_name,
                            "scale": float(obj_dict["scale"]),
                            "value": float(df[obj_name].item()),
                            "weight": float(obj_dict["weight"]),
                        }
                    )

        return obj_values

    @property
    def single_objective_values(self) -> list[dict[str, Any]]:
        batch_datas = pl.concat(
            [
                b.batch_objectives.select(
                    c for c in b.batch_objectives.columns if c != "merit_value"
                ).with_columns(pl.lit(1 if b.is_improvement else 0).alias("accepted"))
                for b in self._ever_storage.data.batches_with_function_results
            ]
        )
        objectives = self._ever_storage.data.objective_functions
        assert objectives is not None
        objective_names = objectives["objective_name"].unique().to_list()

        for o in objectives.to_dicts():
            batch_datas = batch_datas.with_columns(
                pl.col(o["objective_name"]) * o["weight"] / o["scale"]
            )

        columns = [
            "batch",
            "objective",
            "accepted",
            *(objective_names if len(objective_names) > 1 else []),
        ]

        return (
            batch_datas.rename(
                {"total_objective_value": "objective", "batch_id": "batch"}
            )
            .select(columns)
            .to_dicts()
        )

    @property
    def gradient_values(self) -> list[dict[str, Any]]:
        all_batch_data = [
            b.batch_objective_gradient
            for b in self._ever_storage.data.batches_with_gradient_results
            if b.batch_objective_gradient is not None
            and b.is_improvement  # Note: This part might not be sensible
        ]
        if not all_batch_data:
            return []

        all_info = pl.concat(all_batch_data)
        objective_columns = [
            c
            for c in all_info.drop(["batch_id", "control_name"]).columns
            if not c.endswith(".total")
        ]
        return (
            all_info.select("batch_id", "control_name", *objective_columns)
            .unpivot(
                on=objective_columns,
                index=["batch_id", "control_name"],
                variable_name="function",
                value_name="value",
            )
            .rename({"control_name": "control", "batch_id": "batch"})
            .sort(by=["batch", "control"])
            .select(["batch", "function", "control", "value"])
            .to_dicts()
        )

    def summary_values(
        self, batches: list[int] | None = None, keys: list[Any] | None = None
    ) -> pl.DataFrame:
        if batches is None:
            batches = self.batches
        data_frames = []
        assert self._config.storage_dir
        storage = open_storage(self._config.storage_dir, "r")
        experiment = next(storage.experiments)
        for batch_id in batches:
            try:
                ensemble = experiment.get_ensemble_by_name(f"batch_{batch_id}")
                summary = ensemble.load_responses(
                    key="summary",
                    realizations=tuple(self.simulations),
                )
            except (ValueError, KeyError):
                summary = None

            if summary is not None:
                summary = summary.pivot(
                    on="response_key", index=["realization", "time"], sort_columns=True
                )
                # The 'Realization' column exported by ert are
                # the 'simulations' of everest.
                summary = summary.rename({"time": "date", "realization": "simulation"})
                if keys is not None:
                    columns = set(summary.columns).intersection(set(keys))
                    summary = summary[["date", "simulation", *list(columns)]]
                summary = summary.with_columns(
                    pl.Series("batch", [batch_id] * summary.shape[0])
                )

                realization_map = (
                    self._ever_storage.data.simulation_to_geo_realization_map(batch_id)
                )
                realizations = pl.Series(
                    "realization",
                    [realization_map.get(int(sim)) for sim in summary["simulation"]],
                )
                realizations = realizations.cast(pl.Int64, strict=False)
                summary = summary.with_columns(realizations)

                data_frames.append(summary)
        storage.close()
        return pl.concat(data_frames) if data_frames else pl.DataFrame()

    @property
    def output_folder(self) -> str:
        return self._config.output_dir

    def export_dataframes(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        batch_dfs_to_join = {}  # type: ignore
        realization_dfs_to_join = {}  # type: ignore
        perturbation_dfs_to_join = {}  # type: ignore

        batch_ids = [b.batch_id for b in self._ever_storage.data.batches]
        all_controls = (
            self._ever_storage.data.controls["control_name"].to_list()
            if self._ever_storage.data.controls is not None
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

        for batch in self._ever_storage.data.batches:
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
        perturbation_df = pl.concat(perturbation_dfs_to_concat, how="diagonal")

        pert_real_df = pl.concat([realization_df, perturbation_df], how="diagonal")

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

    @property
    def everest_csv(self) -> str:
        export_filename = f"{self._config.config_file}.csv"
        full_path = os.path.join(self.output_folder, export_filename)

        if not os.path.exists(full_path):
            combined_df, _, _ = self.export_dataframes()
            combined_df.write_csv(full_path)

        return full_path
