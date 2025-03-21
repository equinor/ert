import importlib
import logging
import os
from typing import Any

import numpy as np
from pydantic import ValidationError
from ropt.config.enopt import EnOptConfig
from ropt.enums import PerturbationType, VariableType
from ropt.transforms import OptModelTransforms

from everest.config import (
    EverestConfig,
    InputConstraintConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.config.utils import FlattenedControls
from everest.strings import EVEREST


def _parse_controls(controls: FlattenedControls, ropt_config: dict[str, Any]) -> None:
    control_types = [VariableType[type_.upper()] for type_ in controls.types]
    ropt_config["variables"] = {
        "types": None if all(item is None for item in control_types) else control_types,
        "initial_values": controls.initial_guesses,
        "lower_bounds": controls.lower_bounds,
        "upper_bounds": controls.upper_bounds,
        "mask": controls.enabled,
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


def _parse_objectives(
    objective_functions: list[ObjectiveFunctionConfig], ropt_config: dict[str, Any]
) -> None:
    weights: list[float] = []
    function_estimator_indices: list[int] = []
    function_estimators: list = []  # type: ignore

    for objective in objective_functions:
        assert isinstance(objective.name, str)
        weights.append(objective.weight or 1.0)

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
    }
    if function_estimators:
        # Only needed if we specified at least one objective type:
        ropt_config["objectives"]["function_estimators"] = function_estimator_indices
        ropt_config["function_estimators"] = function_estimators


def _get_bounds(
    constraints: list[InputConstraintConfig] | list[OutputConstraintConfig],
) -> tuple[list[float], list[float]]:
    lower_bounds = []
    upper_bounds = []
    for constr in constraints:
        if constr.target is None:
            lower_bounds.append(
                -np.inf if constr.lower_bound is None else constr.lower_bound
            )
            upper_bounds.append(
                np.inf if constr.upper_bound is None else constr.upper_bound
            )
        else:
            if constr.lower_bound is not None or constr.upper_bound is not None:
                raise RuntimeError(
                    "input constraint error: target cannot be combined with bounds"
                )
            lower_bounds.append(constr.target)
            upper_bounds.append(constr.target)
    return lower_bounds, upper_bounds


def _parse_input_constraints(
    input_constraints: list[InputConstraintConfig] | None,
    formatted_control_names: list[str],
    formatted_control_names_dotdash: list[str],
    ropt_config: dict[str, Any],
) -> None:
    def _get_control_index(name: str) -> int:
        try:
            matching_index = formatted_control_names.index(name.replace("-", "."))
            return matching_index
        except ValueError:
            pass

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

        ropt_config["linear_constraints"] = {
            "coefficients": coefficients_matrix,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        }


def _parse_output_constraints(
    output_constraints: list[OutputConstraintConfig] | None, ropt_config: dict[str, Any]
) -> None:
    if output_constraints:
        lower_bounds, upper_bounds = _get_bounds(output_constraints)
        ropt_config["nonlinear_constraints"] = {
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        }


def _parse_optimization(
    ever_opt: OptimizationConfig | None,
    has_output_constraints: bool,
    ropt_config: dict[str, Any],
) -> None:
    ropt_config["optimizer"] = {
        "stdout": "optimizer.stdout",
        "stderr": "optimizer.stderr",
    }
    if not ever_opt:
        return

    ropt_optimizer = ropt_config["optimizer"]
    ropt_gradient = ropt_config["gradient"]

    ropt_optimizer["method"] = ever_opt.algorithm

    if alg_max_iter := ever_opt.max_iterations:
        ropt_optimizer["max_iterations"] = alg_max_iter

    if alg_max_eval := ever_opt.max_function_evaluations:
        ropt_optimizer["max_functions"] = alg_max_eval

    if alg_conv_tol := ever_opt.convergence_tolerance:
        ropt_optimizer["tolerance"] = alg_conv_tol

    if alg_grad_spec := ever_opt.speculative:
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
                ropt_config.get("nonlinear_constraints", {}).get("lower_bounds", [])
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


def _everest2ropt(
    ever_config: EverestConfig, transforms: OptModelTransforms | None
) -> dict[str, Any]:
    """Generate a ropt configuration from an Everest one

    NOTE: This method is a work in progress. So far only the some of
    the values are actually extracted, all the others are set to some
    more or less reasonable default
    """
    ropt_config: dict[str, Any] = {}

    _parse_controls(FlattenedControls(ever_config.controls), ropt_config)
    _parse_objectives(ever_config.objective_functions, ropt_config)
    _parse_input_constraints(
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

    ropt_config["realizations"] = {
        "weights": ever_config.model.realizations_weights,
    }
    if min_real_succ := ever_config.optimization.min_realizations_success:
        ropt_config["realizations"]["realization_min_success"] = min_real_succ

    ropt_config["optimizer"]["output_dir"] = os.path.abspath(
        ever_config.optimization_output_dir
    )
    ropt_config["gradient"]["seed"] = ever_config.environment.random_seed

    return ropt_config


def everest2ropt(
    ever_config: EverestConfig, transforms: OptModelTransforms | None = None
) -> EnOptConfig:
    ropt_dict = _everest2ropt(ever_config, transforms)

    try:
        enopt_config = EnOptConfig.model_validate(ropt_dict, context=transforms)
    except ValidationError as exc:
        ert_version = importlib.metadata.version("ert")
        ropt_version = importlib.metadata.version("ropt")
        msg = (
            f"Validation error(s) in ropt:\n\n{exc}.\n\n"
            "Check the everest installation, there may a be version mismatch.\n"
            f"  (ERT: {ert_version}, ropt: {ropt_version})\n"
            "If the everest installation is correct, please report this as a bug."
        )
        raise ValueError(msg) from exc

    return enopt_config
