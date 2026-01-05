import os
from typing import Any

from ropt.enums import PerturbationType, VariableType

from ert.config import EverestConstraintsConfig, EverestControl, EverestObjectivesConfig
from everest.config import (
    InputConstraintConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.config.utils import get_samplers


def _parse_controls(
    controls: list[EverestControl], random_seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    control_types = [
        VariableType[type_.upper()]
        for control in controls
        for type_ in control.control_types
    ]
    initial_guesses = [
        initial_guess
        for control in controls
        for initial_guess in control.initial_guesses
    ]
    samplers, sampler_indices = get_samplers(controls)
    ropt_variables: dict[str, Any] = {
        "types": None if all(item is None for item in control_types) else control_types,
        "variable_count": len(initial_guesses),
        "lower_bounds": [min_ for control in controls for min_ in control.min],
        "upper_bounds": [max_ for control in controls for max_ in control.max],
        "perturbation_types": [
            PerturbationType[perturbation_type.upper()]
            for control in controls
            for perturbation_type in control.perturbation_types
        ],
        "perturbation_magnitudes": [
            perturbation_magnitude
            for control in controls
            for perturbation_magnitude in control.perturbation_magnitudes
        ],
        "mask": [enabled for control in controls for enabled in control.enabled],
        "seed": random_seed,
        "samplers": sampler_indices,
    }

    ropt_samplers = [
        {}
        if sampler is None
        else {
            "method": sampler.method,
            "options": {} if sampler.options is None else sampler.options,
            "shared": False if sampler.shared is None else sampler.shared,
        }
        for sampler in samplers
    ]

    return ropt_variables, ropt_samplers


def _parse_objectives(
    objective_functions: EverestObjectivesConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weights: list[float] = [
        1.0 if weight is None else weight for weight in objective_functions.weights
    ]
    function_estimator_indices: list[int] = []
    function_estimators: list = []  # type: ignore

    for objective_type in objective_functions.objective_types:
        # If any objective specifies an objective type, we have to specify
        # function estimators in ropt to implement these types. This is done by
        # supplying a list of estimators and for each objective an index into
        # that list:
        function_estimator_idx = next(
            (
                idx
                for idx, estimator in enumerate(function_estimators)
                if estimator["method"] == objective_type
            ),
            None,
        )
        # If not, make a new estimator:
        if function_estimator_idx is None:
            function_estimator_idx = len(function_estimators)
            function_estimators.append({"method": objective_type})
        function_estimator_indices.append(function_estimator_idx)

    ropt_objectives: dict[str, Any] = {"weights": weights}
    ropt_function_estimators: list[dict[str, Any]] = []
    if function_estimators:
        # Only needed if we specified at least one objective type:
        ropt_objectives["function_estimators"] = function_estimator_indices
        ropt_function_estimators = function_estimators

    return ropt_objectives, ropt_function_estimators


def _get_bounds(
    constraints: list[InputConstraintConfig] | list[OutputConstraintConfig],
) -> tuple[list[float], list[float]]:
    lower_bounds = []
    upper_bounds = []
    for constr in constraints:
        if constr.target is None:
            lower_bounds.append(constr.lower_bound)
            upper_bounds.append(constr.upper_bound)
        else:
            lower_bounds.append(constr.target)
            upper_bounds.append(constr.target)
    return lower_bounds, upper_bounds


def _parse_input_constraints(
    input_constraints: list[InputConstraintConfig],
    controls: list[EverestControl],
) -> dict[str, Any]:
    formatted_control_names = [
        name for config in controls for name in config.input_keys
    ]
    formatted_control_names_dotdash = [
        name for config in controls for name in config.input_keys_dotdash
    ]

    def _get_control_index(name: str) -> int:
        try:
            matching_index = formatted_control_names.index(name.replace("-", "."))
        except ValueError:
            pass
        else:
            return matching_index

        # Dash is deprecated, should eventually be removed
        # along with formatted_control_names_dotdash
        return formatted_control_names_dotdash.index(name)

    if input_constraints:
        coefficients_matrix = []
        for constr in input_constraints:
            coefficients = [0.0] * len(formatted_control_names)
            for name, value in constr.weights.items():
                index = _get_control_index(name)
                coefficients[index] = value
            coefficients_matrix.append(coefficients)

        lower_bounds, upper_bounds = _get_bounds(input_constraints)

        return {
            "coefficients": coefficients_matrix,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        }
    return {}


def _parse_output_constraints(
    output_constraints: EverestConstraintsConfig | None,
) -> dict[str, Any]:
    if output_constraints:
        return {
            "lower_bounds": [
                target if target is not None else lb
                for lb, target in zip(
                    output_constraints.lower_bounds,
                    output_constraints.targets,
                    strict=False,
                )
            ],
            "upper_bounds": [
                target if target is not None else ub
                for ub, target in zip(
                    output_constraints.upper_bounds,
                    output_constraints.targets,
                    strict=False,
                )
            ],
        }
    return {}


def _parse_optimization(
    ever_opt: OptimizationConfig | None,
    realizations_weights: list[float],
    has_output_constraints: bool,
    optimization_output_dir: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ropt_optimizer: dict[str, Any] = {
        "output_dir": os.path.abspath(optimization_output_dir),
        "stdout": "optimizer.stdout",
        "stderr": "optimizer.stderr",
    }
    ropt_gradient: dict[str, Any] = {}
    ropt_realizations: dict[str, Any] = {"weights": realizations_weights}
    cvar_config: dict[str, Any] = {}

    if ever_opt is not None:
        ropt_optimizer["method"] = ever_opt.algorithm

        if ever_opt.max_iterations is not None:
            ropt_optimizer["max_iterations"] = ever_opt.max_iterations

        if ever_opt.max_function_evaluations is not None:
            ropt_optimizer["max_functions"] = ever_opt.max_function_evaluations

        if ever_opt.max_batch_num is not None:
            ropt_optimizer["max_batches"] = ever_opt.max_batch_num

        if ever_opt.convergence_tolerance is not None:
            ropt_optimizer["tolerance"] = ever_opt.convergence_tolerance

        if ever_opt.speculative:
            ropt_gradient["evaluation_policy"] = "speculative"

        options: list[str] | dict[str, Any] = []
        if ever_opt.options is not None:
            options = ever_opt.options
        elif ever_opt.backend_options is not None:
            options = ever_opt.backend_options

        # The constraint_tolerance option is only used by Dakota:
        if (
            has_output_constraints
            and ever_opt.constraint_tolerance is not None
            and ever_opt.optimization_plugin_name == "dakota"
            and (
                ever_opt.algorithm
                in {"conmin_mfd", "conmin_frcg", "asynch_pattern_search"}
            )
        ):
            assert isinstance(options, list)
            options += [f"constraint_tolerance = {ever_opt.constraint_tolerance}"]

        if options:
            ropt_optimizer["options"] = options

        ropt_optimizer["parallel"] = ever_opt.parallel

        if ever_opt.perturbation_num is not None:
            ropt_gradient["number_of_perturbations"] = ever_opt.perturbation_num
            # For a single perturbation, use the ensemble for gradient calculation:
            ropt_gradient["merge_realizations"] = (
                ropt_gradient["number_of_perturbations"] == 1
            )

        if ever_opt.min_pert_success is not None:
            ropt_gradient["perturbation_min_success"] = ever_opt.min_pert_success

        if ever_opt.min_realizations_success is not None:
            ropt_realizations["realization_min_success"] = (
                ever_opt.min_realizations_success
            )

        if (cvar_opts := ever_opt.cvar) is not None:
            # set up the configuration of the realization filter that implements cvar:
            if cvar_opts.percentile is not None:
                cvar_config = {
                    "method": "cvar-objective",
                    "options": {
                        "percentile": cvar_opts.percentile,
                    },
                }
            elif cvar_opts.number_of_realizations is not None:
                cvar_config = {
                    "method": "sort-objective",
                    "options": {
                        "first": 0,
                        "last": cvar_opts.number_of_realizations - 1,
                    },
                }

    return ropt_optimizer, ropt_gradient, ropt_realizations, cvar_config


def everest2ropt(
    controls: list[EverestControl],
    objective_functions: EverestObjectivesConfig,
    input_constraints: list[InputConstraintConfig],
    output_constraints: EverestConstraintsConfig | None,
    optimization: OptimizationConfig | None,
    model: ModelConfig,
    random_seed: int,
    optimization_output_dir: str,
) -> tuple[dict[str, Any], list[float]]:
    ropt_variables, ropt_samplers = _parse_controls(controls, random_seed)
    ropt_objectives, ropt_function_estimators = _parse_objectives(objective_functions)
    ropt_linear_constraints = _parse_input_constraints(input_constraints, controls)
    ropt_nonlinear_constraints = _parse_output_constraints(output_constraints)
    ropt_optimizer, ropt_gradient, ropt_realizations, cvar_config = _parse_optimization(
        ever_opt=optimization,
        realizations_weights=model.realizations_weights,
        has_output_constraints=bool(output_constraints),
        optimization_output_dir=optimization_output_dir,
    )
    ropt_realization_filters: list[dict[str, Any]] = []

    if cvar_config:
        # Both objective and constraint configurations use an array of
        # indices to any realization filters that should be applied. In this
        # case, we want all objectives and constraints to refer to the same
        # filter implementing cvar:
        objective_count = len(ropt_objectives["weights"])
        ropt_objectives["realization_filters"] = objective_count * [0]
        if ropt_nonlinear_constraints:
            constraint_count = len(ropt_nonlinear_constraints["lower_bounds"])
            ropt_nonlinear_constraints["realization_filters"] = constraint_count * [0]
        cvar_config["options"].update({"sort": list(range(objective_count))})
        ropt_realization_filters = [cvar_config]
        # For efficiency, function and gradient evaluations should be split
        # so that no unnecessary gradients are calculated:
        ropt_gradient["evaluation_policy"] = "separate"

    ropt_config: dict[str, Any] = {
        "variables": ropt_variables,
        "objectives": ropt_objectives,
        "realizations": ropt_realizations,
        "optimizer": ropt_optimizer,
        "names": {
            "variable": [name for config in controls for name in config.input_keys],
            "objective": objective_functions.keys,
            "nonlinear_constraint": output_constraints.keys
            if output_constraints is not None
            else [],
            "realization": model.realizations,
        },
    }
    if ropt_linear_constraints:
        ropt_config["linear_constraints"] = ropt_linear_constraints
    if ropt_nonlinear_constraints:
        ropt_config["nonlinear_constraints"] = ropt_nonlinear_constraints
    if ropt_gradient:
        ropt_config["gradient"] = ropt_gradient
    if ropt_realization_filters:
        ropt_config["realization_filters"] = ropt_realization_filters
    if ropt_function_estimators:
        ropt_config["function_estimators"] = ropt_function_estimators
    if ropt_samplers:
        ropt_config["samplers"] = ropt_samplers

    return ropt_config, [
        initial_guess
        for control in controls
        for initial_guess in control.initial_guesses
    ]
