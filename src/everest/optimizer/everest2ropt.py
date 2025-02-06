import logging
import os
from typing import Any

from ropt.config.enopt import EnOptConfig, EnOptContext
from ropt.enums import ConstraintType, PerturbationType, VariableType
from ropt.transforms import OptModelTransforms

from everest.config import (
    EverestConfig,
    InputConstraintConfig,
    ModelConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.config.utils import FlattenedControls
from everest.strings import EVEREST


def _parse_controls(controls: FlattenedControls, ropt_config):
    control_types = [
        None if type_ is None else VariableType[type_.upper()]
        for type_ in controls.types
    ]
    indices = [idx for idx, is_enabled in enumerate(controls.enabled) if is_enabled]
    ropt_config["variables"] = {
        "types": None if all(item is None for item in control_types) else control_types,
        "initial_values": controls.initial_guesses,
        "lower_bounds": controls.lower_bounds,
        "upper_bounds": controls.upper_bounds,
        "indices": indices if indices else None,
    }

    if "gradients" not in ropt_config:
        ropt_config["gradient"] = {}

    if any(item >= 0 for item in controls.sampler_indices):
        ropt_config["samplers"] = [
            {
                "method": sampler.method,
                "options": {} if sampler.options is None else sampler.options,
                "shared": False if sampler.shared is None else sampler.shared,
            }
            for sampler in controls.samplers
        ]
        ropt_config["gradient"]["samplers"] = controls.sampler_indices

    default_magnitude = (max(controls.upper_bounds) - min(controls.lower_bounds)) / 10.0
    ropt_config["gradient"]["perturbation_magnitudes"] = [
        default_magnitude if perturbation_magnitude is None else perturbation_magnitude
        for perturbation_magnitude in controls.perturbation_magnitudes
    ]

    ropt_config["gradient"]["perturbation_types"] = [
        PerturbationType[perturbation_type.upper()]
        for perturbation_type in controls.perturbation_types
    ]


def _parse_objectives(objective_functions: list[ObjectiveFunctionConfig], ropt_config):
    scales: list[float] = []
    auto_scale: list[bool] = []
    weights: list[float] = []
    function_estimator_indices: list[int] = []
    function_estimators: list = []

    for objective in objective_functions:
        assert isinstance(objective.name, str)
        weights.append(objective.weight or 1.0)
        scales.append(objective.scale or 1.0)
        auto_scale.append(objective.auto_scale or False)

        # If any objective specifies an objective type, we have to specify
        # function estimators in ropt to implement these types. This is done by
        # supplying a list of estimators and for each objective an index into
        # that list:
        objective_type = objective.type
        if objective_type is None:
            objective_type = "mean"
        # Find the estimator if it exists:
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

    ropt_config["objectives"] = {
        "weights": weights,
        "scales": scales,
        "auto_scale": auto_scale,
    }
    if function_estimators:
        # Only needed if we specified at least one objective type:
        ropt_config["objectives"]["function_estimators"] = function_estimator_indices
        ropt_config["function_estimators"] = function_estimators


def _parse_input_constraints(
    controls: FlattenedControls,
    input_constraints: list[InputConstraintConfig] | None,
    formatted_control_names: list[str],
    formatted_control_names_dotdash: list[str],
    ropt_config,
):
    if not input_constraints:
        return

    def _get_control_index(name: str):
        try:
            matching_index = formatted_control_names.index(name.replace("-", "."))
            return matching_index
        except ValueError:
            pass

        # Dash is deprecated, should eventually be removed
        # along with formatted_control_names_dotdash
        return formatted_control_names_dotdash.index(name)

    coefficients_matrix = []
    rhs_values = []
    types = []

    def _add_input_constraint(rhs_value, coefficients, constraint_type):
        if rhs_value is not None:
            coefficients_matrix.append(coefficients)
            rhs_values.append(rhs_value)
            types.append(constraint_type)

    for constr in input_constraints:
        coefficients = [0.0] * len(formatted_control_names)
        for name, value in constr.weights.items():
            index = _get_control_index(name)
            coefficients[index] = value

        target = constr.target
        upper_bound = constr.upper_bound
        lower_bound = constr.lower_bound
        if target is not None and (upper_bound is not None or lower_bound is not None):
            raise RuntimeError(
                "input constraint error: target cannot be combined with bounds"
            )
        _add_input_constraint(target, coefficients, ConstraintType.EQ)
        _add_input_constraint(upper_bound, coefficients, ConstraintType.LE)
        _add_input_constraint(lower_bound, coefficients, ConstraintType.GE)

    ropt_config["linear_constraints"] = {
        "coefficients": coefficients_matrix,
        "rhs_values": rhs_values,
        "types": types,
    }


def _parse_output_constraints(
    output_constraints: list[OutputConstraintConfig] | None, ropt_config
):
    if not output_constraints:
        return

    rhs_values: list[float] = []
    scales: list[float] = []
    auto_scale: list[bool] = []
    types: list[ConstraintType] = []

    def _add_output_constraint(
        rhs_value: float | None, constraint_type: ConstraintType, suffix=None
    ):
        if rhs_value is not None:
            rhs_values.append(rhs_value)
            scales.append(constr.scale if constr.scale is not None else 1.0)
            auto_scale.append(constr.auto_scale or False)
            types.append(constraint_type)

    for constr in output_constraints:
        target = constr.target
        upper_bound = constr.upper_bound
        lower_bound = constr.lower_bound
        if target is not None and (upper_bound is not None or lower_bound is not None):
            raise RuntimeError(
                "output constraint error: target cannot be combined with bounds"
            )
        _add_output_constraint(
            target,
            ConstraintType.EQ,
        )
        _add_output_constraint(
            upper_bound,
            ConstraintType.LE,
            None if lower_bound is None else "upper",
        )
        _add_output_constraint(
            lower_bound,
            ConstraintType.GE,
            None if upper_bound is None else "lower",
        )

    ropt_config["nonlinear_constraints"] = {
        "rhs_values": rhs_values,
        "scales": scales,
        "auto_scale": auto_scale,
        "types": types,
    }


def _parse_optimization(
    ever_opt: OptimizationConfig | None,
    has_output_constraints: bool,
    ropt_config,
):
    ropt_config["optimizer"] = {}
    if not ever_opt:
        return

    ropt_optimizer = ropt_config["optimizer"]
    ropt_gradient = ropt_config["gradient"]

    algorithm = ever_opt.algorithm or "optpp_q_newton"
    ropt_optimizer["method"] = f"{algorithm}"

    alg_max_iter = ever_opt.max_iterations
    if alg_max_iter:
        ropt_optimizer["max_iterations"] = alg_max_iter

    alg_max_eval = ever_opt.max_function_evaluations
    if alg_max_eval:
        ropt_optimizer["max_functions"] = alg_max_eval

    alg_conv_tol = ever_opt.convergence_tolerance or None
    if alg_conv_tol:
        ropt_optimizer["tolerance"] = alg_conv_tol

    alg_grad_spec = ever_opt.speculative or None
    if alg_grad_spec:
        ropt_optimizer["speculative"] = alg_grad_spec

    # Handle the backend options. Due to historical reasons there two keywords:
    # "options" is used to pass a list of string, "backend_options" is used to
    # pass a dict. These are redirected to the same ropt option:
    if ever_opt.backend_options is not None:
        message = (
            "optimization.backend_options is deprecated. "
            "Please use optimization.options instead, "
            "it will accept both objects and lists of strings."
        )
        print(message)
        logging.getLogger(EVEREST).warning(message)

    options = ever_opt.options or ever_opt.backend_options or {}

    alg_const_tol = ever_opt.constraint_tolerance or None
    if (
        has_output_constraints
        and alg_const_tol is not None
        and isinstance(options, list)
    ):
        options += [f"constraint_tolerance = {alg_const_tol}"]

    # The constraint_tolerance option is only used by Dakota:
    ropt_optimizer["options"] = options

    parallel = True if ever_opt.parallel is None else ever_opt.parallel
    ropt_optimizer["parallel"] = True if parallel is None else parallel

    if ever_opt.perturbation_num is not None:
        ropt_gradient["number_of_perturbations"] = ever_opt.perturbation_num
        # For a single perturbation, use the ensemble for gradient calculation:
        ropt_gradient["merge_realizations"] = (
            ropt_gradient["number_of_perturbations"] == 1
        )

    min_per_succ = ever_opt.min_pert_success
    if min_per_succ is not None:
        ropt_gradient["perturbation_min_success"] = min_per_succ

    if cvar_opts := ever_opt.cvar or None:
        # set up the configuration of the realization filter that implements cvar:
        if (percentile := cvar_opts.percentile) is not None:
            cvar_config: dict[str, Any] = {
                "method": "cvar-objective",
                "options": {"percentile": percentile},
            }
        elif (realizations := cvar_opts.number_of_realizations) is not None:
            cvar_config = {
                "method": "sort-objective",
                "options": {"first": 0, "last": realizations - 1},
            }
        else:
            cvar_config = {}

        if cvar_config:
            # Both objective and constraint configurations use an array of
            # indices to any realization filters that should be applied. In this
            # case, we want all objectives and constraints to refer to the same
            # filter implementing cvar:
            objective_count = len(ropt_config["objectives"]["weights"])
            constraint_count = len(
                ropt_config.get("nonlinear_constraints", {}).get("rhs_values", [])
            )
            ropt_config["objectives"]["realization_filters"] = objective_count * [0]
            if constraint_count > 0:
                ropt_config["nonlinear_constraints"]["realization_filters"] = (
                    constraint_count * [0]
                )
            cvar_config["options"].update({"sort": list(range(objective_count))})
            ropt_config["realization_filters"] = [cvar_config]
            # For efficiency, function and gradient evaluations should be split
            # so that no unnecessary gradients are calculated:
            ropt_optimizer["split_evaluations"] = True


def _parse_model(
    ever_model: ModelConfig | None,
    ever_opt: OptimizationConfig | None,
    ropt_config,
):
    if not ever_model:
        return

    ever_reals = ever_model.realizations or []
    ever_reals_weights = ever_model.realizations_weights
    if ever_reals_weights is None:
        ever_reals_weights = [1.0 / len(ever_reals)] * len(ever_reals)

    ropt_config["realizations"] = {
        "weights": ever_reals_weights,
    }
    min_real_succ = ever_opt.min_realizations_success if ever_opt else None
    if min_real_succ is not None:
        ropt_config["realizations"]["realization_min_success"] = min_real_succ


def _parse_environment(
    optimization_output_dir: str, random_seed: int | None, ropt_config
):
    ropt_config["optimizer"]["output_dir"] = os.path.abspath(optimization_output_dir)
    if random_seed is not None:
        ropt_config["gradient"]["seed"] = random_seed


def everest2ropt(
    ever_config: EverestConfig, transforms: OptModelTransforms | None = None
) -> EnOptConfig:
    """Generate a ropt configuration from an Everest one

    NOTE: This method is a work in progress. So far only the some of
    the values are actually extracted, all the others are set to some
    more or less reasonable default
    """
    ropt_config: dict[str, Any] = {}

    flattened_controls = FlattenedControls(ever_config.controls)

    _parse_controls(flattened_controls, ropt_config)
    _parse_objectives(ever_config.objective_functions, ropt_config)
    _parse_input_constraints(
        flattened_controls,
        ever_config.input_constraints,
        ever_config.formatted_control_names,
        ever_config.formatted_control_names_dotdash,
        ropt_config,
    )
    _parse_output_constraints(ever_config.output_constraints, ropt_config)
    _parse_optimization(
        ever_opt=ever_config.optimization,
        has_output_constraints=ever_config.output_constraints is not None,
        ropt_config=ropt_config,
    )
    _parse_model(
        ever_model=ever_config.model,
        ever_opt=ever_config.optimization,
        ropt_config=ropt_config,
    )
    _parse_environment(
        optimization_output_dir=ever_config.optimization_output_dir,
        random_seed=ever_config.environment.random_seed
        if ever_config.environment
        else None,
        ropt_config=ropt_config,
    )

    return EnOptConfig.model_validate(
        ropt_config,
        context=None if transforms is None else EnOptContext(transforms=transforms),
    )
