from collections import OrderedDict

import polars as pl
from seba_sqlite.snapshot import SebaSnapshot

from ert.storage import open_storage
from everest.config import EverestConfig, ServerConfig
from everest.detached import ServerStatus, everserver_status


class EverestDataAPI:
    def __init__(self, config: EverestConfig, filter_out_gradient=True):
        self._config = config
        output_folder = config.optimization_output_dir
        self._snapshot = SebaSnapshot(output_folder).get_snapshot(filter_out_gradient)

    @property
    def batches(self):
        batch_ids = list({opt.batch_id for opt in self._snapshot.optimization_data})
        return sorted(batch_ids)

    @property
    def accepted_batches(self):
        batch_ids = list(
            {opt.batch_id for opt in self._snapshot.optimization_data if opt.merit_flag}
        )
        return sorted(batch_ids)

    @property
    def objective_function_names(self):
        return [fnc.name for fnc in self._snapshot.metadata.objectives.values()]

    @property
    def output_constraint_names(self):
        return [fnc.name for fnc in self._snapshot.metadata.constraints.values()]

    def input_constraint(self, control):
        controls = [
            con
            for con in self._snapshot.metadata.controls.values()
            if con.name == control
        ]
        return {"min": controls[0].min_value, "max": controls[0].max_value}

    def output_constraint(self, constraint):
        """
        :return: a dictionary with two keys: "type" and "right_hand_side".
                 "type" has three options:
                     ["lower_bound", "upper_bound", "target"]
                 "right_hand_side" is a constant real number that indicates
                 the constraint bound/target.
        """
        constraints = [
            con
            for con in self._snapshot.metadata.constraints.values()
            if con.name == constraint
        ]
        return {
            "type": constraints[0].constraint_type,
            "right_hand_side": constraints[0].rhs_value,
        }

    @property
    def realizations(self):
        return list(
            OrderedDict.fromkeys(
                int(sim.realization) for sim in self._snapshot.simulation_data
            )
        )

    @property
    def simulations(self):
        return list(
            OrderedDict.fromkeys(
                [int(sim.simulation) for sim in self._snapshot.simulation_data]
            )
        )

    @property
    def control_names(self):
        return [con.name for con in self._snapshot.metadata.controls.values()]

    @property
    def control_values(self):
        controls = [con.name for con in self._snapshot.metadata.controls.values()]
        return [
            {"control": con, "batch": sim.batch, "value": sim.controls[con]}
            for sim in self._snapshot.simulation_data
            for con in controls
            if con in sim.controls
        ]

    @property
    def objective_values(self):
        return [
            {
                "function": objective.name,
                "batch": sim.batch,
                "realization": sim.realization,
                "simulation": sim.simulation,
                "value": sim.objectives[objective.name],
                "weight": objective.weight,
                "norm": objective.normalization,
            }
            for sim in self._snapshot.simulation_data
            for objective in self._snapshot.metadata.objectives.values()
            if objective.name in sim.objectives
        ]

    @property
    def single_objective_values(self):
        single_obj = [
            {
                "batch": optimization_el.batch_id,
                "objective": optimization_el.objective_value,
                "accepted": optimization_el.merit_flag,
            }
            for optimization_el in self._snapshot.optimization_data
        ]
        metadata = {
            func.name: {"weight": func.weight, "norm": func.normalization}
            for func in self._snapshot.metadata.functions.values()
            if func.function_type == func.FUNCTION_OBJECTIVE_TYPE
        }
        if len(metadata) == 1:
            return single_obj
        objectives = []
        for name, values in self._snapshot.expected_objectives.items():
            for idx, val in enumerate(values):
                factor = metadata[name]["weight"] * metadata[name]["norm"]
                if len(objectives) > idx:
                    objectives[idx].update({name: val * factor})
                else:
                    objectives.append({name: val * factor})
        for idx, obj in enumerate(single_obj):
            obj.update(objectives[idx])

        return single_obj

    @property
    def gradient_values(self):
        return [
            {
                "batch": optimization_el.batch_id,
                "function": function,
                "control": control,
                "value": value,
            }
            for optimization_el in self._snapshot.optimization_data
            for function, info in optimization_el.gradient_info.items()
            for control, value in info.items()
        ]

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
                # The realization ID as defined by Everest must be
                # retrieved via the seba snapshot.
                realization_map = {
                    sim.simulation: sim.realization
                    for sim in self._snapshot.simulation_data
                    if sim.batch == batch_id
                }
                realizations = pl.Series(
                    "realization",
                    [realization_map.get(str(sim)) for sim in summary["simulation"]],
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
