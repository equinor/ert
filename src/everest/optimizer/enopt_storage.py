# Copyright (C) The Netherlands Organisation for Applied Scientific Research,
# TNO, 2015-2022. All rights reserved.
#
# This file is part of Seba: a proprietary software library for ensemble based
# optimization developed by TNO. This file, the Seba software or data or
# information contained in the software may not be copied or distributed without
# prior written permission from TNO.
#
# Seba and the information and data contained in this software are confidential.
# Neither the whole or any part of the software and the data and information it
# contains may be disclosed to any third party without the prior written consent
# of The Netherlands Organisation for Applied Scientific Research (TNO).
import dataclasses
from typing import Any, Dict, List, Optional
from uuid import UUID

from ropt.enums import EventType
from ropt.optimization import Event
from ropt.plan import OptimizationPlanRunner
from ropt.results import FunctionResults, GradientResults
from seba_sqlite import Function

from ert.storage import Experiment, Storage


@dataclasses.dataclass
class OptimizationInfo:
    controls: Dict[str, str]
    objective_value: float
    merit_flag: int
    batch_id: int
    gradient_info: Dict[str, Dict[str, float]]


@dataclasses.dataclass
class Metadata:
    realizations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    functions: Dict[str, Any] = dataclasses.field(default_factory=dict)
    controls: Dict[str, Any] = dataclasses.field(default_factory=dict)


# OptimizationInfo = namedtuple(
#    "OptimizationInfo", "controls objective_value merit_flag batch_id gradient_info"
# )
# Contains information about the optimization steps:
#   controls        ->  {control_name : control_value}
#   objective_value -> value # objective value of the optimization step
#   merit_flag      -> 0 or 1 # 1 if the optimization step increases the merit
#                      of the optimization process
#   batch_id        -> the id of the batch generating the optimization step
#   gradient_info   -> gradient information per function for each of the controls.
#   Ex: {
#   'function_name_1': {'control_name_1': control_value,
#                       'control_name_2': control_value,},
#   'function_name_2': {'control_name_1': control_value,
#                       'control_name_2': control_value,}
#   }

# Metadata = namedtuple(
#    "Metadata", "realizations functions objectives constraints controls"
# )
# Contains information about the Seba DB definitions:
#   realization -> dictionary mapping realization ids to seba Realization
#                  objects
#   functions   -> dictionary mapping function ids to Seba Function objects
#   controls    -> dictionary mapping control ids to Seba ControlDefinition
#                  objects


@dataclasses.dataclass
class SimulationInfo:
    batch: str
    objectives: Dict[str, Any]
    constraints: Dict[str, Any]
    controls: Dict[str, Any]
    sim_avg_obj: float
    is_gradient: bool
    realization: str
    start_time: Any
    end_time: Any
    success: bool
    realization_weight: float
    simulation: Any


# SimulationInfo = namedtuple(
#    "SimulationInfo",
#    "batch objectives constraints controls sim_avg_obj"
#    " is_gradient realization start_time end_time success"
#    " realization_weight simulation",
# )
# Contains information about the simulations:
#     batch -> the batch id
#     objectives  -> Dictionary mapping the objective function names to the
#                    objective values per simulation also contains mapping
#                    of the normalized and weighted normalized objective values
#     constraints -> Dictionary mapping the constraint function names to the
#                    constraint values per simulation also contains mapping of
#                    the normalized and weighted normalized constraint values
#     controls    -> Dictionary mapping the control names to their values.
#                    Controls generating the simulation results
#     sim_avg_obj -> The value of the objective function for the simulation
#     is_gradient -> Flag describing if the simulation is a gradient or non
#                    gradient simulation
#     realization -> The name of the realization the simulation is part of
#     start_time  -> The starting timpestamp for the simulation
#     end_time    -> The end timpstamp for the simulation
#     success     -> Flag describing if the simulation was successful or not (1 or 0)
#     realization_weight -> The weight of the realization the simulation was part of.
#     simulation  -> The simulation number used in libres


@dataclasses.dataclass
class EnOptStorageROPTObserver:
    enopt_storage: "EnOptStorage"
    experiment: Optional[Experiment] = None

    def observe_optimization(self, runner: OptimizationPlanRunner):
        runner.add_observer(
            EventType.FINISHED_EVALUATION,
            self._handle_optimization_finished_batch_event,
        )

        runner.add_observer(
            EventType.FINISHED_OPTIMIZER_STEP,
            self._handle_optimization_finish_event,
        )

    def _handle_optimization_finished_batch_event(
        self, event: Event, experiment: Experiment
    ):
        print("h")

        for result in event.results:
            if isinstance(result, GradientResults):
                print("gradient!")
            elif isinstance(result, FunctionResults):
                print("function!")
            else:
                print("???")

            print("aa")

    def _handle_optimization_finish_event(self, event: Event, experiment: Experiment):
        print("h")


@dataclasses.dataclass
class EnOptStorage:
    metadata: Metadata = dataclasses.field(default_factory=Metadata)
    simulation_data: List[SimulationInfo] = dataclasses.field(default_factory=list)
    optimization_data: List[OptimizationInfo] = dataclasses.field(default_factory=list)

    def create_ropt_observer(self) -> EnOptStorageROPTObserver:
        return EnOptStorageROPTObserver(enopt_storage=self)

    def observe_optimizer(
        self, optimizer: OptimizationPlanRunner, storage: Storage, experiment_id: UUID
    ):
        experiment = storage.get_experiment(experiment_id)

        optimizer.add_observer(
            EventType.FINISHED_EVALUATION,
            lambda event: self._handle_optimization_finished_batch_event(
                event, experiment
            ),
        )

        optimizer.add_observer(
            EventType.FINISHED_OPTIMIZER_STEP,
            lambda event: self._handle_optimization_finish_event(event, experiment),
        )

    @property
    def expected_single_objective(self) -> Optional[List[float]]:
        # The list of optimization values for each of the optimization steps. In
        # the case of multiple realizations for each optimization step the value
        # is computed as the sum of the optimization value multiplied with the
        # realization weight
        if self.optimization_data:
            return [data_row.objective_value for data_row in self._optimization_data]

        return None

    @property
    def optimization_data_by_batch(self) -> Optional[Dict[str, OptimizationInfo]]:
        # :return: A dictionary mapping the batch id o the OptimizationInfo element
        if self.optimization_data:
            return {
                optimization.batch_id: optimization
                for optimization in self._optimization_data
            }
        return None

    @property
    def expected_objectives(self) -> Optional[Dict[str, float]]:
        # Constructs a dictionary mapping the objective function names to the
        # respective objective value per optimization step. If we are dealing with
        # multiple realization the objective value is calculated as the weighted
        # sum of the individual objective value function per each realization.
        # :return: Expected objective function dictionary.
        # Ex: {
        #      'function_name_1': [obj_val_1, obj_val_1, ..],
        #      'function_name_2': [obj_val_1, obj_val_1, ..]
        #      }
        objective_names = [func.name for func in self.metadata.objectives.values()]
        constraint_names = [func.name for func in self.metadata.constraints.values()]
        function_names = objective_names + constraint_names
        if not self.simulation_data:
            return None

        first_realization = self.simulation_data[0].realization
        expected_objectives = {}
        for sim_info in self.simulation_data:
            for function_name in sim_info.objectives:
                if function_name not in function_names:
                    continue
                if function_name in expected_objectives:
                    if sim_info.realization != first_realization:
                        expected_objectives[function_name][-1] += (
                            sim_info.objectives[function_name]
                            * sim_info.realization_weight
                        )
                    else:
                        expected_objectives[function_name].append(
                            sim_info.objectives[function_name]
                            * sim_info.realization_weight
                        )
                else:
                    expected_objectives[function_name] = [
                        sim_info.objectives[function_name] * sim_info.realization_weight
                    ]
        return expected_objectives

    @property
    def increased_merit_flags(self) -> Optional[List[int]]:
        # :return: A list of merit flags. Ex: [1, 1, 0, 1, 0, 1]
        if self.optimization_data:
            return [data_row.merit_flag for data_row in self.optimization_data]

        return None

    @property
    def increased_merit_indices(self) -> Optional[List[int]]:
        # The index of the single expected objected function that provides an
        # improvement in the optimization process
        # :return: list of indices ex: [0, 1, 3, 5]
        if self.increased_merit_flags:
            return [
                index
                for index, flag in enumerate(self.increased_merit_flags)
                if flag > 0
            ]
        return None

    @property
    def optimization_controls(self) -> Optional[Dict[str, List[float]]]:
        # Controls for each of the optimization steps
        # :return: dictionary mapping the control names with a list of control
        # values for each of the optimization steps
        if not self.optimization_data:
            return None

        optimization_controls = {}
        for data in self.optimization_data:
            for name, value in data.controls.items():
                if name in optimization_controls:
                    optimization_controls[name].append(value)
                else:
                    optimization_controls[name] = [value]

        return optimization_controls


class OptimizationSnapshot:
    def __init__(self):
        self._data = None

    def _simulations_data(self, simulations):
        # Constructs a list of SimulationInfo elements based on a given list of
        # simulations.
        # :param simulations: List of simulation used to construct the
        # SimulationInfo elements
        # :return: A list of SimulationInfo elements
        functions_info = {
            func.function_id: func for func in self.database.load_functions()
        }
        realization_info = {
            realization.realization_id: realization
            for realization in self.database.load_realizations()
        }

        simulation_data = []

        for sim in simulations:
            sim_results = self.database.load_simulation_results(
                simulation_id=sim.simulation_id
            )
            objectives = {}
            constraints = {}
            sim_avg = 0.0
            realization = realization_info[sim.realization_id]

            for result in sim_results:
                function = functions_info[result.function_id]
                if function.function_type == Function.FUNCTION_OBJECTIVE_TYPE:
                    objectives[function.name] = result.value
                    objectives[function.name + "_norm"] = (
                        result.value * function.normalization
                    )
                    objectives[function.name + "_weighted_norm"] = (
                        result.value * function.normalization * function.weight
                    )
                    sim_avg += objectives[function.name + "_weighted_norm"]
                else:
                    constraints[function.name] = result.value
                    constraints[function.name + "_norm"] = (
                        result.value * function.normalization
                    )
                    constraints[function.name + "_weighted_norm"] = (
                        result.value * function.normalization * function.weight
                    )

            simulation_element = SimulationInfo(
                batch=int(sim.batch_id),
                objectives=objectives,
                constraints=constraints,
                controls=dict(self.database.load_control_values(set_id=sim.set_id)),
                sim_avg_obj=sim_avg,
                is_gradient=sim.is_gradient,
                realization=realization.name,
                start_time=sim.start_time_stamp,
                end_time=sim.end_time_stamp,
                success=sim.success,
                realization_weight=realization.weight,
                simulation=sim.name.split("_")[-1],
            )

            simulation_data.append(simulation_element)
        return simulation_data

    def _gradients_for_batch_id(self, batch_id, func_info, control_info):
        # Retrieves gradient information for a specified bach id.
        # :param batch_id: The required batch id
        # :param func_info: dictionary mapping function ids to the Function
        # objects in the Seba DB
        # :param control_info: dictionary mapping control definition ids to the
        # ControlDefinitions in the Seba DB
        # :return: Gradient information per function for each of the controls.
        # Returned dictionary example structure
        # {
        #  'function_name_1': {'control_name_1': control_value,
        #                      'control_name_2': control_value,},
        #  'function_name_2': {'control_name_1': control_value,
        #                      'control_name_2': control_value,}
        # }
        gradient_info = {}
        for result in self.database.load_gradient_results(batch_id=batch_id):
            function = func_info[result.function_id]
            control = control_info[result.control_definition_id]
            if function.name in gradient_info:
                gradient_info[function.name].update({control.name: result.value})
            else:
                gradient_info[function.name] = {control.name: result.value}
        return gradient_info

    def _optimization_data(self):
        # :return: List of optimizationInfo elements
        func_info = {func.function_id: func for func in self.database.load_functions()}
        control_info = {
            control.control_id: control
            for control in self.database.load_control_definitions()
        }
        calc_results = self.database.load_calculation_results()
        calc_results.sort(key=lambda x: x.batch_id)
        optimization_data = []

        for result in calc_results:
            gradient_info = self._gradients_for_batch_id(
                result.batch_id, func_info, control_info
            )
            optimization_data.append(
                OptimizationInfo(
                    controls=dict(
                        self.database.load_control_values(set_id=result.set_id)
                    ),
                    objective_value=result.object_function_value,
                    merit_flag=result.improved_flag,
                    batch_id=result.batch_id,
                    gradient_info=gradient_info,
                )
            )

        return optimization_data

    def _metadata(self):
        # :return: The metadata data structure
        return Metadata(
            realizations={
                realization.realization_id: realization
                for realization in self.database.load_realizations()
            },
            functions={
                func.function_id: func for func in self.database.load_functions()
            },
            objectives={
                func.function_id: func
                for func in self.database.load_functions(
                    function_type=Function.FUNCTION_OBJECTIVE_TYPE
                )
            },
            constraints={
                func.function_id: func
                for func in self.database.load_functions(
                    function_type=Function.FUNCTION_CONSTRAINT_TYPE
                )
            },
            controls={
                control.control_id: control
                for control in self.database.load_control_definitions()
            },
        )

    def _filtered_simulation_data(self, filter_out_gradient=False, batches=None):
        # Filter simulation by gradient type and by bach ids
        # :param filter_out_gradient: If set to true will only return a list of
        # non gradient simulation
        # :param batches: List of batch ids. If given the return list will contain
        # only simulations associated with the given batch ids
        # :return: List of filtered simulations
        simulations = self.database.load_simulations(
            filter_out_gradient=filter_out_gradient
        )
        if batches:
            simulations = [sim for sim in simulations if sim.batch_id in batches]
        return simulations
