import datetime
import time
from collections import defaultdict
from itertools import count
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy import float64
from numpy._typing import NDArray
from ropt.evaluator import Evaluator, EvaluatorContext, EvaluatorResult

from ert import BatchSimulator, WorkflowRunner
from ert.config import ErtConfig, HookRuntime
from ert.storage import Storage
from everest.config import EverestConfig


class Simulator(BatchSimulator):
    """Everest simulator: BatchSimulator"""

    def __init__(
        self,
        ever_config: EverestConfig,
        ert_config: ErtConfig,
        storage: Storage,
        callback=None,
    ) -> None:
        experiment = storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",
            parameters=ert_config.ensemble_config.parameter_configuration,
            responses=ert_config.ensemble_config.response_configuration,
        )

        super(Simulator, self).__init__(
            experiment=experiment,
            perferred_num_cpu=ert_config.preferred_num_cpu,
            user_config_file=ert_config.user_config_file,
            env_vars=ert_config.env_vars,
            forward_model_steps=ert_config.forward_model_steps,
            runpath_file=str(ert_config.runpath_file),
            parameter_configurations=ert_config.ensemble_config.parameter_configs,
            queue_config=ert_config.queue_config,
            model_config=ert_config.model_config,
            analysis_config=ert_config.analysis_config,
            hooked_workflows=ert_config.hooked_workflows,
            substitutions=ert_config.substitutions,
            templates=ert_config.ert_templates,
            controls=self._get_controls(ever_config),
            results=self._get_results(ever_config),
            callback=callback,
        )

        self._function_aliases = self._get_aliases(ever_config)
        self._experiment_id = None
        self._batch = 0
        self._cache: Optional[_SimulatorCache] = None
        if ever_config.simulator is not None and ever_config.simulator.enable_cache:
            self._cache = _SimulatorCache()

        self.storage = storage

    def _get_controls(self, ever_config: EverestConfig) -> List[str]:
        controls = ever_config.controls or []
        return [control.name for control in controls]

    def _get_results(self, ever_config: EverestConfig) -> List[str]:
        objectives_names = [
            objective.name
            for objective in ever_config.objective_functions
            if objective.alias is None
        ]
        constraint_names = [
            constraint.name for constraint in (ever_config.output_constraints or [])
        ]
        return objectives_names + constraint_names

    def _get_aliases(self, ever_config: EverestConfig) -> Dict[str, str]:
        aliases = {
            objective.name: objective.alias
            for objective in ever_config.objective_functions
            if objective.alias is not None
        }
        constraints = ever_config.output_constraints or []
        for constraint in constraints:
            if (
                constraint.upper_bound is not None
                and constraint.lower_bound is not None
            ):
                aliases[f"{constraint.name}:lower"] = constraint.name
                aliases[f"{constraint.name}:upper"] = constraint.name
        return aliases

    def create_forward_model_evaluator_function(
        self,
    ) -> Evaluator:
        def run_forward_model(
            control_values: NDArray[np.float64], metadata: EvaluatorContext
        ) -> EvaluatorResult:
            nonlocal self
            return self._run_forward_model(control_values, metadata)

        return run_forward_model

    def _run_forward_model(
        self, control_values: NDArray[np.float64], metadata: EvaluatorContext
    ) -> EvaluatorResult:
        active = (
            np.ones(control_values.shape[0], dtype=np.bool_)
            if metadata.active is None
            else np.fromiter(
                (metadata.active[realization] for realization in metadata.realizations),
                dtype=np.bool_,
            )
        )
        case_data = []
        cached = {}
        assert metadata.config.realizations.names is not None
        realization_ids = [
            metadata.config.realizations.names[realization]  # type: ignore
            for realization in metadata.realizations
        ]

        for sim_idx, real_id in enumerate(realization_ids):
            if self._cache is not None:
                cache_id = self._cache.find_key(real_id, control_values[sim_idx, :])
                if cache_id is not None:
                    cached[sim_idx] = cache_id
                    active[sim_idx] = False

            if active[sim_idx]:
                controls: DefaultDict[str, Any] = defaultdict(dict)
                assert metadata.config.variables.names is not None
                for control_name, control_value in zip(
                    metadata.config.variables.names,  # type: ignore
                    control_values[sim_idx, :],
                ):
                    self._add_control(controls, control_name, control_value)
                case_data.append((real_id, controls))

        sim_context = self.start(f"batch_{self._batch}", case_data)

        while sim_context.running():
            time.sleep(0.2)
        results = sim_context.results()

        # Pre-simulation workflows are run by sim_context, but
        # post-stimulation workflows are not, do it here:
        ensemble = sim_context.get_ensemble()
        for workflow in self.hooked_workflows[HookRuntime.POST_SIMULATION]:
            WorkflowRunner(
                workflow,
                self.storage,
                ensemble,
            ).run_blocking()

        for fnc_name, alias in self._function_aliases.items():
            for result in results:
                result[fnc_name] = result[alias]

        names = metadata.config.objective_functions.names
        objectives = self._get_active_results(
            results,
            names,  # type: ignore
            control_values,
            active,
        )

        constraints = None
        if metadata.config.nonlinear_constraints is not None:
            names = metadata.config.nonlinear_constraints.names
            assert names is not None
            constraints = self._get_active_results(
                results,  # type: ignore
                names,  # type: ignore
                control_values,
                active,
            )

        if self._cache is not None:
            for sim_idx, cache_id in cached.items():
                objectives[sim_idx, ...] = self._cache.get_objectives(cache_id)
                if constraints is not None:
                    constraints[sim_idx, ...] = self._cache.get_constraints(cache_id)

        sim_ids = np.empty(control_values.shape[0], dtype=np.intc)
        sim_ids.fill(-1)
        sim_ids[active] = np.arange(len(results), dtype=np.intc)

        # Note the negative sign for the objective results. Everest aims to do a
        # maximization, while the standard practice of minimizing is followed by
        # ropt. Therefore we will minimize the negative of the objectives:
        result = EvaluatorResult(
            batch_id=self._batch,
            objectives=-objectives,
            constraints=constraints,
            evaluation_ids=sim_ids,
        )

        # Add the results from active simulations to the cache:
        if self._cache is not None:
            for sim_idx, real_id in enumerate(realization_ids):
                if active[sim_idx]:
                    self._cache.add_simulation_results(
                        sim_idx, real_id, control_values, objectives, constraints
                    )

        self._batch += 1
        return result

    @staticmethod
    def _add_control(
        controls: Mapping[str, Any], control_name: Tuple[Any, ...], control_value: float
    ) -> None:
        group_name = control_name[0]
        variable_name = control_name[1]
        group = controls[group_name]
        if len(control_name) > 2:
            index_name = str(control_name[2])
            if variable_name in group:
                group[variable_name][index_name] = control_value
            else:
                group[variable_name] = {index_name: control_value}
        else:
            group[variable_name] = control_value

    @staticmethod
    def _get_active_results(
        results: List[Dict[str, NDArray[np.float64]]],
        names: Tuple[str],
        controls: NDArray[np.float64],
        active: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        values = np.zeros((controls.shape[0], len(names)), dtype=float64)
        for func_idx, name in enumerate(names):
            values[active, func_idx] = np.fromiter(
                (np.nan if result is None else result[name][0] for result in results),
                dtype=np.float64,
            )
        return values

    @property
    def number_of_evaluated_batches(self) -> int:
        return self._batch


# This cache can be used to prevent re-evaluation of forward models. Due to its
# simplicity it has some limitations:
#   - There is no limit on the number of cached entries.
#   - Searching in the cache is by brute-force, iterating over the entries.
# Both of these should not be an issue for the intended use with cases where the
# forward models are very expensive to compute: The number of cached entries is
# not expected to become prohibitively large.
class _SimulatorCache:
    def __init__(self) -> None:
        # Stores the realization/controls key, together with an ID.
        self._keys: DefaultDict[int, List[Tuple[NDArray[np.float64], int]]] = (
            defaultdict(list)
        )
        # Store objectives and constraints by ID:
        self._objectives: Dict[int, NDArray[np.float64]] = {}
        self._constraints: Dict[int, NDArray[np.float64]] = {}

        # Generate unique ID's:
        self._counter = count()

    def add_simulation_results(
        self,
        sim_idx: int,
        real_id: int,
        control_values: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: Optional[NDArray[np.float64]],
    ):
        cache_id = next(self._counter)
        self._keys[real_id].append((control_values[sim_idx, :].copy(), cache_id))
        self._objectives[cache_id] = objectives[sim_idx, ...].copy()
        if constraints is not None:
            self._constraints[cache_id] = constraints[sim_idx, ...].copy()

    def find_key(
        self, real_id: int, control_vector: NDArray[np.float64]
    ) -> Optional[int]:
        # Brute-force search, premature optimization is the root of all evil:
        for cached_vector, cache_id in self._keys.get(real_id, []):
            if np.allclose(
                control_vector,
                cached_vector,
                rtol=0.0,
                atol=float(np.finfo(np.float32).eps),
            ):
                return cache_id
        return None

    def get_objectives(self, cache_id: int) -> NDArray[np.float64]:
        return self._objectives[cache_id]

    def get_constraints(self, cache_id: int) -> NDArray[np.float64]:
        return self._constraints[cache_id]
