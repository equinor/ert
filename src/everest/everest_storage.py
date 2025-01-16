import copy
import logging
import os
import sqlite3
import time
from collections import namedtuple
from itertools import count
from pathlib import Path

import numpy as np
from ropt.enums import ConstraintType, EventType
from ropt.results import FunctionResults, GradientResults, convert_to_maximize

from .database import Database
from .snapshot import SebaSnapshot

OptimalResult = namedtuple(
    "OptimalResult", "batch, controls, total_objective, expected_objectives"
)

logger = logging.getLogger(__name__)


def _convert_names(control_names):
    converted_names = []
    for name in control_names:
        converted = f"{name[0]}_{name[1]}"
        if len(name) > 2:
            converted += f"-{name[2]}"
        converted_names.append(converted)
    return converted_names


class EverestStorage:
    # This implementation builds as much as possible on the older database and
    # snapshot code, since it is meant for backwards compatibility, and should
    # not be extended with new functionality.

    def __init__(self, optimizer, output_dir):
        # Internal variables.
        self._output_dir = output_dir
        self._database = Database(output_dir)
        self._control_ensemble_id = 0
        self._gradient_ensemble_id = 0
        self._simulator_results = None
        self._merit_file = Path(output_dir) / "dakota" / "OPT_DEFAULT.out"

        # Connect event handlers.
        self._set_event_handlers(optimizer)

        self._initialized = False

    @property
    def file(self):
        return self._database.location

    def _initialize(self, event):
        if self._initialized:
            return
        self._initialized = True

        self._database.add_experiment(
            name="optimization_experiment", start_time_stamp=time.time()
        )

        # Add configuration values.
        config = event.config
        for control_name, initial_value, lower_bound, upper_bound in zip(
            _convert_names(config.variables.names),
            config.variables.initial_values,
            config.variables.lower_bounds,
            config.variables.upper_bounds,
            strict=False,
        ):
            self._database.add_control_definition(
                control_name, initial_value, lower_bound, upper_bound
            )

        for name, weight, scale in zip(
            config.objective_functions.names,
            config.objective_functions.weights,
            config.objective_functions.scales,
            strict=False,
        ):
            self._database.add_function(
                name=name,
                function_type="OBJECTIVE",
                weight=weight,
                normalization=1.0 / scale,
            )

        if config.nonlinear_constraints is not None:
            for name, scale, rhs_value, constraint_type in zip(
                config.nonlinear_constraints.names,
                config.nonlinear_constraints.scales,
                config.nonlinear_constraints.rhs_values,
                config.nonlinear_constraints.types,
                strict=False,
            ):
                self._database.add_function(
                    name=name,
                    function_type="CONSTRAINT",
                    normalization=scale,
                    rhs_value=rhs_value,
                    constraint_type=ConstraintType(constraint_type).name.lower(),
                )

        for name, weight in zip(
            config.realizations.names,
            config.realizations.weights,
            strict=False,
        ):
            self._database.add_realization(str(name), weight)

    def _add_batch(self, config, controls, perturbed_controls):
        self._gradient_ensemble_id += 1
        self._control_ensemble_id = self._gradient_ensemble_id
        control_names = _convert_names(config.variables.names)
        for control_name, value in zip(control_names, controls, strict=False):
            self._database.add_control_value(
                set_id=self._control_ensemble_id,
                control_name=control_name,
                value=value,
            )
        if perturbed_controls is not None:
            perturbed_controls = perturbed_controls.reshape(
                perturbed_controls.shape[0], -1
            )
            self._gradient_ensemble_id = self._control_ensemble_id
            for g_idx in range(perturbed_controls.shape[1]):
                self._gradient_ensemble_id += 1
                for c_idx, c_name in enumerate(control_names):
                    self._database.add_control_value(
                        set_id=self._gradient_ensemble_id,
                        control_name=c_name,
                        value=perturbed_controls[c_idx, g_idx],
                    )

    def _add_simulations(self, config, result):
        self._gradient_ensemble_id = self._control_ensemble_id
        simulation_index = count()
        if isinstance(result, FunctionResults):
            for realization_name in config.realizations.names:
                self._database.add_simulation(
                    realization_name=str(realization_name),
                    set_id=self._control_ensemble_id,
                    sim_name=f"{result.batch_id}_{next(simulation_index)}",
                    is_gradient=False,
                )
        if isinstance(result, GradientResults):
            for realization_name in config.realizations.names:
                for _ in range(config.gradient.number_of_perturbations):
                    self._gradient_ensemble_id += 1
                    self._database.add_simulation(
                        realization_name=str(realization_name),
                        set_id=self._gradient_ensemble_id,
                        sim_name=f"{result.batch_id}_{next(simulation_index)}",
                        is_gradient=True,
                    )

    def _add_simulator_results(
        self, config, batch, objective_results, constraint_results
    ):
        if constraint_results is None:
            results = objective_results
        else:
            results = np.vstack((objective_results, constraint_results))
        statuses = np.logical_and.reduce(np.isfinite(results), axis=0)
        names = config.objective_functions.names
        if config.nonlinear_constraints is not None:
            names += config.nonlinear_constraints.names

        for sim_idx, status in enumerate(statuses):
            sim_name = f"{batch}_{sim_idx}"
            for function_idx, name in enumerate(names):
                if status:
                    self._database.add_simulation_result(
                        sim_name, results[function_idx, sim_idx], name, 0
                    )
            self._database.set_simulation_ended(sim_name, status)

    def _add_constraint_values(self, config, batch, constraint_values):
        statuses = np.logical_and.reduce(np.isfinite(constraint_values), axis=0)
        for sim_id, status in enumerate(statuses):
            if status:
                for idx, constraint_name in enumerate(
                    config.nonlinear_constraints.names
                ):
                    # Note the time_index=0, the database supports storing
                    # multipel time-points, but we do not support that, so we
                    # use times_index=0.
                    self._database.update_simulation_result(
                        simulation_name=f"{batch}_{sim_id}",
                        function_name=constraint_name,
                        times_index=0,
                        value=constraint_values[idx, sim_id],
                    )

    def _add_gradients(self, config, objective_gradients):
        for grad_index, gradient in enumerate(objective_gradients):
            for control_index, control_name in enumerate(
                _convert_names(config.variables.names)
            ):
                self._database.add_gradient_result(
                    gradient[control_index],
                    config.objective_functions.names[grad_index],
                    1,
                    control_name,
                )

    def _add_total_objective(self, total_objective):
        self._database.add_calculation_result(
            set_id=self._control_ensemble_id,
            object_function_value=total_objective,
        )

    def _convert_constraints(self, config, constraint_results):
        constraint_results = copy.deepcopy(constraint_results)
        rhs_values = config.nonlinear_constraints.rhs_values
        for idx, constraint_type in enumerate(config.nonlinear_constraints.types):
            constraint_results[idx] -= rhs_values[idx]
            if constraint_type == ConstraintType.LE:
                constraint_results[idx] *= -1.0
        return constraint_results

    def _store_results(self, config, results):
        if isinstance(results, FunctionResults):
            objective_results = results.evaluations.objectives
            objective_results = np.moveaxis(objective_results, -1, 0)
            constraint_results = results.evaluations.constraints
            if constraint_results is not None:
                constraint_results = np.moveaxis(constraint_results, -1, 0)
        else:
            objective_results = None
            constraint_results = None

        if isinstance(results, GradientResults):
            perturbed_variables = results.evaluations.perturbed_variables
            perturbed_variables = np.moveaxis(perturbed_variables, -1, 0)
            perturbed_objectives = results.evaluations.perturbed_objectives
            perturbed_objectives = np.moveaxis(perturbed_objectives, -1, 0)
            perturbed_constraints = results.evaluations.perturbed_constraints
            if perturbed_constraints is not None:
                perturbed_constraints = np.moveaxis(perturbed_constraints, -1, 0)
        else:
            perturbed_variables = None
            perturbed_objectives = None
            perturbed_constraints = None

        self._add_batch(config, results.evaluations.variables, perturbed_variables)
        self._add_simulations(config, results)

        # Convert back the simulation results to the legacy format:
        if perturbed_objectives is not None:
            perturbed_objectives = perturbed_objectives.reshape(
                perturbed_objectives.shape[0], -1
            )
            if objective_results is None:
                objective_results = perturbed_objectives
            else:
                objective_results = np.hstack((objective_results, perturbed_objectives))

        if config.nonlinear_constraints is not None:
            if perturbed_constraints is not None:
                perturbed_constraints = perturbed_constraints.reshape(
                    perturbed_constraints.shape[0], -1
                )
                if constraint_results is None:
                    constraint_results = perturbed_constraints
                else:
                    constraint_results = np.hstack(
                        (constraint_results, perturbed_constraints)
                    )
            # The legacy code converts all constraints to the form f(x) >=0:
            constraint_results = self._convert_constraints(config, constraint_results)

        self._add_simulator_results(
            config, results.batch_id, objective_results, constraint_results
        )
        if config.nonlinear_constraints:
            self._add_constraint_values(config, results.batch_id, constraint_results)
        if isinstance(results, FunctionResults) and results.functions is not None:
            self._add_total_objective(results.functions.weighted_objective)
        if isinstance(results, GradientResults) and results.gradients is not None:
            self._add_gradients(config, results.gradients.objectives)

    def _handle_finished_batch_event(self, event):
        logger.debug("Storing batch results in the sqlite database")

        converted_results = tuple(
            convert_to_maximize(result) for result in event.results
        )
        results = []
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
            results = [best_results] + results
        last_batch = -1
        for item in results:
            if item.batch_id != last_batch:
                self._database.add_batch()
            self._store_results(event.config, item)
            if item.batch_id != last_batch:
                self._database.set_batch_ended
            last_batch = item.batch_id

        self._database.set_batch_ended(time.time(), True)

        # Merit values are dakota specific, load them if the output file exists:
        self._database.update_calculation_result(_get_merit_values(self._merit_file))

        backup_data(self._database.location)

    def _handle_finished_event(self, event):
        logger.debug("Storing final results in the sqlite database")
        self._database.update_calculation_result(_get_merit_values(self._merit_file))
        self._database.set_experiment_ended(time.time())

    def _set_event_handlers(self, optimizer):
        optimizer.add_observer(EventType.START_OPTIMIZER_STEP, self._initialize)
        optimizer.add_observer(
            EventType.FINISHED_EVALUATION, self._handle_finished_batch_event
        )
        optimizer.add_observer(
            EventType.FINISHED_OPTIMIZER_STEP,
            self._handle_finished_event,
        )

    def get_optimal_result(self):
        snapshot = SebaSnapshot(self._output_dir)
        optimum = next(
            (
                data
                for data in reversed(snapshot.get_optimization_data())
                if data.merit_flag
            ),
            None,
        )
        if optimum is None:
            return None
        objectives = snapshot.get_snapshot(batches=[optimum.batch_id])
        return OptimalResult(
            batch=optimum.batch_id,
            controls=optimum.controls,
            total_objective=optimum.objective_value,
            expected_objectives={
                name: value[0] for name, value in objectives.expected_objectives.items()
            },
        )


def backup_data(database_location) -> None:
    src = sqlite3.connect(database_location)
    dst = sqlite3.connect(database_location + ".backup")
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


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


def _get_merit_values(merit_file):
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
    if merit_file.exists():
        for line in _get_merit_fn_lines(merit_file):
            value = _parse_merit_line(line)
            if value is not None:
                merit_values.append({"iter": value[0], "value": value[1]})
    return merit_values
