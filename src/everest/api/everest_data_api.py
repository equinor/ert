import os
from pathlib import Path
from typing import Any

import polars as pl

from ert.storage import open_storage
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage


class EverestDataAPI:
    def __init__(self, config: EverestConfig, filter_out_gradient: bool = True) -> None:
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

    @property
    def everest_csv(self) -> str:
        return str(self._ever_storage.export_everest_opt_results_to_csv())
