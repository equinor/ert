import os
from pathlib import Path

import polars
import polars as pl
from ropt.enums import ConstraintType

from ert.storage import open_storage
from everest.config import EverestConfig, ServerConfig
from everest.detached import ServerStatus, everserver_status
from everest.everest_storage import EverestStorage


class EverestDataAPI:
    def __init__(self, config: EverestConfig, filter_out_gradient=True):
        self._config = config
        output_folder = config.optimization_output_dir
        self._ever_storage = EverestStorage(Path(output_folder))

        if os.path.exists(output_folder):
            self._ever_storage.read_from_output_dir()

    @property
    def batches(self):
        return sorted(
            b.batch_id
            for b in self._ever_storage.data.batches
            if b.batch_objectives is not None
        )

    @property
    def accepted_batches(self):
        return sorted(
            b.batch_id for b in self._ever_storage.data.batches if b.is_improvement
        )

    @property
    def objective_function_names(self):
        return sorted(
            self._ever_storage.data.objective_functions["objective_name"]
            .unique()
            .to_list()
        )

    @property
    def output_constraint_names(self):
        return (
            sorted(
                self._ever_storage.data.nonlinear_constraints["constraint_name"]
                .unique()
                .to_list()
            )
            if self._ever_storage.data.nonlinear_constraints is not None
            else []
        )

    def input_constraint(self, control):
        # Note: This function is weird, its existence is probably not well-justified
        # consider removing!
        initial_values = self._ever_storage.data.controls
        control_spec = initial_values.filter(
            pl.col("control_name") == control
        ).to_dicts()[0]
        return {
            "min": control_spec.get("lower_bounds"),
            "max": control_spec.get("upper_bounds"),
        }

    def output_constraint(self, constraint):
        """
        :return: a dictionary with two keys: "type" and "right_hand_side".
                 "type" has three options:
                     ["lower_bound", "upper_bound", "target"]
                 "right_hand_side" is a constant real number that indicates
                 the constraint bound/target.
        """

        constraint_dict = self._ever_storage.data.nonlinear_constraints.filter(
            polars.col("constraint_name") == constraint
        ).to_dicts()[0]
        return {
            "type": ConstraintType(constraint_dict["constraint_type"]).name.lower(),
            "right_hand_side": constraint_dict["constraint_rhs_value"],
        }

    @property
    def realizations(self):
        return sorted(
            self._ever_storage.data.batches[0]
            .realization_objectives["realization"]
            .unique()
            .to_list()
        )

    @property
    def simulations(self):
        return sorted(
            self._ever_storage.data.batches[0]
            .realization_objectives["simulation_id"]
            .unique()
            .to_list()
        )

    @property
    def control_names(self):
        return sorted(
            self._ever_storage.data.controls["control_name"].unique().to_list()
        )

    @property
    def control_values(self):
        all_control_names = self._ever_storage.data.controls["control_name"].to_list()
        new = []
        for batch in self._ever_storage.data.batches:
            if batch.realization_controls is None:
                continue

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
    def objective_values(self):
        obj_values = []
        for b in self._ever_storage.data.batches:
            if b.realization_objectives is None:
                continue

            for (
                geo_realization,
                simulation_id,
            ), df in b.realization_objectives.sort(
                ["realization", "simulation_id"]
            ).group_by(["realization", "simulation_id"], maintain_order=True):
                for obj_dict in self._ever_storage.data.objective_functions.sort(
                    ["objective_name"]
                ).to_dicts():
                    obj_name = obj_dict["objective_name"]
                    obj_values.append(
                        {
                            "batch": int(b.batch_id),
                            "realization": int(geo_realization),
                            "simulation": int(simulation_id),
                            "function": obj_name,
                            "norm": float(obj_dict["normalization"]),
                            "value": float(df[obj_name].item()),
                            "weight": float(obj_dict["weight"]),
                        }
                    )

        return obj_values

    @property
    def single_objective_values(self):
        batch_datas = polars.concat(
            [
                b.batch_objectives.select(
                    c for c in b.batch_objectives.columns if c != "merit_value"
                ).with_columns(
                    polars.lit(1 if b.is_improvement else 0).alias("accepted")
                )
                for b in self._ever_storage.data.batches
                if b.realization_controls is not None
            ]
        )
        objectives = self._ever_storage.data.objective_functions
        objective_names = objectives["objective_name"].unique().to_list()

        for o in objectives.to_dicts():
            batch_datas = batch_datas.with_columns(
                polars.col(o["objective_name"]) * o["weight"] * o["normalization"]
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
    def gradient_values(self):
        all_batch_data = [
            b.batch_objective_gradient
            for b in self._ever_storage.data.batches
            if b.batch_objective_gradient is not None and b.is_improvement
        ]
        if not all_batch_data:
            return []

        all_info = polars.concat(all_batch_data)
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

    def summary_values(self, batches=None, keys=None):
        if batches is None:
            batches = self.batches
        data_frames = []
        storage = open_storage(self._config.storage_dir, "r")
        experiment = next(storage.experiments)
        for batch_id in batches:
            ensemble = experiment.get_ensemble_by_name(f"batch_{batch_id}")
            try:
                summary = ensemble.load_responses(
                    key="summary",
                    realizations=tuple(self.simulations),
                )
            except (ValueError, KeyError):
                summary = pl.DataFrame()

            if not summary.is_empty():
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
        return pl.concat(data_frames)

    @property
    def output_folder(self):
        return self._config.output_dir

    @property
    def everest_csv(self):
        status_path = ServerConfig.get_everserver_status_path(self._config.output_dir)
        state = everserver_status(status_path)
        if state["status"] == ServerStatus.completed:
            return self._config.export_path
        else:
            return None
