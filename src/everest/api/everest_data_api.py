import logging
from typing import Any

import polars as pl

from ert.storage import open_storage
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage

logger = logging.getLogger(__name__)


class EverestDataAPI:
    def __init__(self, config: EverestConfig, filter_out_gradient: bool = True) -> None:
        self._config = config
        assert config.storage_dir.exists()
        self._experiment = EverestStorage.get_everest_experiment(config.storage_dir)

    @property
    def batches(self) -> list[int]:
        return sorted(
            ensemble.iteration
            for ensemble in self._experiment.ensembles_with_function_results
        )

    @property
    def accepted_batches(self) -> list[int]:
        return sorted(
            ensemble.iteration
            for ensemble in sorted(
                self._experiment.ensembles, key=lambda ens: ens.iteration
            )
            if ensemble.is_improvement
        )

    @property
    def objective_function_names(self) -> list[str]:
        return self._experiment.objective_functions.keys

    @property
    def output_constraint_names(self) -> list[str]:
        constraints_config = self._experiment.output_constraints
        if not constraints_config:
            return []

        return constraints_config.keys

    @property
    def realizations(self) -> list[int]:
        if not self._experiment.ensembles_with_function_results:
            return []

        realization_objectives = self._experiment.ensembles_with_function_results[
            0
        ].realization_objectives

        assert realization_objectives is not None
        return sorted(realization_objectives["realization"].unique().to_list())

    @property
    def simulations(self) -> list[int]:
        if not self._experiment.ensembles_with_function_results:
            return []

        realization_objectives = self._experiment.ensembles_with_function_results[
            0
        ].realization_objectives

        assert realization_objectives is not None
        return sorted(realization_objectives["simulation_id"].unique().to_list())

    @property
    def control_names(self) -> list[str]:
        return self._experiment.parameter_keys

    @property
    def control_values(self) -> list[dict[str, Any]]:
        all_control_names = self._experiment.parameter_keys

        new = []
        for ensemble in self._experiment.ensembles_with_function_results:
            assert ensemble.realization_controls is not None
            for controls_dict in ensemble.realization_controls.to_dicts():
                for name in all_control_names:
                    new.append(
                        {
                            "control": name,
                            "batch": ensemble.iteration,
                            "value": controls_dict[name],
                        }
                    )

        return new

    @property
    def objective_values(self) -> list[dict[str, Any]]:
        obj_values = []

        objectives = self._experiment.objective_functions
        for ensemble in self._experiment.ensembles_with_function_results:
            assert ensemble.realization_objectives is not None
            for (
                model_realization,
                simulation_id,
            ), df in ensemble.realization_objectives.sort(
                ["realization", "simulation_id"]
            ).group_by(["realization", "simulation_id"], maintain_order=True):
                for key, scale, weight in zip(
                    objectives.keys, objectives.scales, objectives.weights, strict=False
                ):
                    obj_value = float(df[key].item())
                    if obj_value is None:
                        logger.error(
                            f"Objective {key} has no value for "
                            f"batch {ensemble.iteration}, "
                            f"model realization {model_realization},"
                            f"simulation id {simulation_id}. "
                            f"Columns in dataframe: {', '.join(df.columns)}"
                        )
                        continue

                    obj_values.append(
                        {
                            "batch": int(ensemble.iteration),
                            "realization": int(model_realization),
                            "simulation": int(simulation_id),
                            "function": key,
                            "scale": float(scale) if scale is not None else None,
                            "value": obj_value,
                            "weight": float(weight) if weight is not None else None,
                        }
                    )

        return obj_values

    @property
    def single_objective_values(self) -> list[dict[str, Any]]:
        batch_datas = pl.concat(
            [
                ens.batch_objectives.select(
                    c for c in ens.batch_objectives.columns if c != "merit_value"
                ).with_columns(pl.lit(1 if ens.is_improvement else 0).alias("accepted"))
                for ens in self._experiment.ensembles_with_function_results
                if ens.batch_objectives is not None  # <-- skip None
            ]
        )
        objectives = self._experiment.objective_functions
        assert objectives is not None

        for name, weight, scale in zip(
            objectives.keys, objectives.weights, objectives.scales, strict=False
        ):
            batch_datas = batch_datas.with_columns(pl.col(name) * weight / scale)
        columns = [
            "batch",
            "objective",
            "accepted",
            *(objectives.keys if len(objectives.keys) > 1 else []),
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
            ensemble.batch_objective_gradient
            for ensemble in self._experiment.ensembles_with_gradient_results
            if ensemble.batch_objective_gradient is not None
            and ensemble.is_improvement  # Note: This part might not be sensible
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
        identical_columns_in_all_batches: bool = True
        summary_columns: list[str] | None = None
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
                if summary_columns is None:
                    summary_columns = summary.columns
                identical_columns_in_all_batches &= summary_columns == summary.columns

                # The 'Realization' column exported by ert are
                # the 'simulations' of everest.
                summary = summary.rename({"time": "date", "realization": "simulation"})
                if keys is not None:
                    columns = set(summary.columns).intersection(set(keys))
                    summary = summary[["date", "simulation", *list(columns)]]
                summary = summary.with_columns(
                    pl.Series("batch", [batch_id] * summary.shape[0])
                )

                ensemble = self._experiment.get_ensemble_by_name(f"batch_{batch_id}")
                realization_info = ensemble._index.everest_realization_info
                assert realization_info is not None

                model_realization_map = {
                    realization: info["model_realization"]
                    for realization, info in realization_info.items()
                }
                realizations = pl.Series(
                    "realization",
                    [
                        model_realization_map.get(int(sim))
                        for sim in summary["simulation"]
                    ],
                )
                realizations = realizations.cast(pl.Int64, strict=False)
                summary = summary.with_columns(realizations)

                data_frames.append(summary)
        storage.close()
        return (
            pl.concat(
                data_frames,
                how="vertical" if identical_columns_in_all_batches else "diagonal",
            )
            if data_frames
            else pl.DataFrame()
        )

    @property
    def output_folder(self) -> str:
        return self._config.output_dir

    @property
    def everest_csv(self) -> str:
        return str(self._experiment.export_everest_opt_results_to_csv())
