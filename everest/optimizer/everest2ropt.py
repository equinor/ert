import os
from typing import Any, Dict, List, Optional

from ropt.enums import ConstraintType, PerturbationType, VariableType

from everest.config import EverestConfig
from everest.config.sampler_config import SamplerConfig


def _parse_sampler(ever_sampler: SamplerConfig):
    sampler_keys = {
        "backend": "backend",
        "backend_options": "options",
        "method": "method",
        "shared": "shared",
    }
    return {
        sampler_keys[key]: value
        for key, value in ever_sampler.model_dump(exclude_none=True).items()
        if key in sampler_keys and value is not None
    }


def _parse_controls(ever_config: EverestConfig, ropt_config):
    """Extract info from ever_config['controls']"""
    names = []
    control_types = []
    initial_values = []
    min_values = []
    max_values = []
    scales = []
    offsets = []
    magnitudes = []
    pert_types = []
    have_auto_scale = False

    for group in ever_config.controls:
        sampler = group.sampler
        if sampler is not None:
            g_sampler = _parse_sampler(sampler)
            g_sampler["control_names"] = []
            if "samplers" not in ropt_config:
                ropt_config["samplers"] = [g_sampler]
            else:
                ropt_config["samplers"].append(g_sampler)
        else:
            g_sampler = None

        g_magnitudes = []
        for control in group.variables:
            # Collect info about the controls
            cname = control.name
            c_index = control.index
            ropt_control_name = (
                (group.name, cname) if c_index is None else (group.name, cname, c_index)
            )
            names.append(ropt_control_name)
            initial_values.append(
                control.initial_guess
                if control.initial_guess is not None
                else group.initial_guess
            )
            control_type = (
                control.control_type
                if control.control_type is not None
                else group.control_type
            )
            control_types.append(
                VariableType.REAL
                if control_type is None
                else VariableType[control_type.upper()]
            )

            auto_scale = control.auto_scale or group.auto_scale
            scaled_range = control.scaled_range or group.scaled_range or [0, 1.0]
            cmin = control.min if control.min is not None else group.min
            cmax = control.max if control.max is not None else group.max

            # Naked asserts to pacify mypy
            assert cmin is not None
            assert cmax is not None

            min_values.append(cmin)
            max_values.append(cmax)
            if auto_scale:
                have_auto_scale = True
                scale = (cmax - cmin) / (scaled_range[1] - scaled_range[0])
                offset = cmin - scaled_range[0] * scale
                scales.append(scale)
                offsets.append(offset)
            else:
                scales.append(1.0)
                offsets.append(0.0)

            pert_mag = control.perturbation_magnitude
            g_magnitudes.append(pert_mag)
            pert_types.append(
                PerturbationType.SCALED
                if auto_scale
                else PerturbationType[(group.perturbation_type or "absolute").upper()]
            )

            sampler = control.sampler
            if sampler is not None:
                c_sampler = _parse_sampler(sampler)
                c_sampler["control_names"] = [ropt_control_name]
                if "samplers" not in ropt_config:
                    ropt_config["samplers"] = [c_sampler]
                else:
                    ropt_config["samplers"].append(c_sampler)
            elif g_sampler is not None:
                g_sampler["control_names"].append(ropt_control_name)

        default_pert_mag = (max(max_values) - min(min_values)) / 10.0
        g_pert_mag = (
            group.perturbation_magnitude
            if group.perturbation_magnitude is not None
            else default_pert_mag
        )
        magnitudes += [g_pert_mag if mag is None else mag for mag in g_magnitudes]

    ropt_config["variables"] = {
        "names": names,
        "types": (
            None
            if all(item == VariableType.REAL for item in control_types)
            else control_types
        ),
        "initial_values": initial_values,
        "lower_bounds": min_values,
        "upper_bounds": max_values,
        "scales": scales if have_auto_scale else None,
        "offsets": offsets if have_auto_scale else None,
        "delimiters": ".-",
    }
    ropt_config["gradient"] = {
        "perturbation_magnitudes": magnitudes,
        "perturbation_types": pert_types,
    }

    # The samplers in the list constructed above contain the names of the
    # variables they should apply to, but ropt expects a array of indices that
    # map variables to the samplers that should apply to them:
    if ropt_config.get("samplers", []):
        sampler_indices = [0] * len(names)
        for idx, sampler in enumerate(ropt_config.get("samplers", [])):
            assert isinstance(sampler, dict)
            for name in sampler["control_names"]:
                sampler_indices[names.index(name)] = idx
            del sampler["control_names"]
        ropt_config["gradient"]["samplers"] = sampler_indices


def _parse_objectives(ever_config: EverestConfig, ropt_config):
    names: List[str] = []
    scales: List[float] = []
    auto_scale: List[bool] = []
    weights: List[float] = []
    transform_indices: List[int] = []
    transforms: List = []

    ever_objs = ever_config.objective_functions or []
    for objective in ever_objs:
        assert isinstance(objective.name, str)
        names.append(objective.name)
        weights.append(objective.weight or 1.0)
        scales.append(1.0 / (objective.normalization or 1.0))
        auto_scale.append(objective.auto_normalize or False)

        # If any objective specifies an objective type, we have to specify
        # function transforms in ropt to implement these types. This is done by
        # supplying a list of transforms and for each objective an index into
        # that list:
        objective_type = objective.type
        if objective_type is None:
            objective_type = "mean"
        # Find the transform if it exists:
        transform_idx = next(
            (
                idx
                for idx, transform in enumerate(transforms)
                if transform["method"] == objective_type
            ),
            None,
        )
        # If not, make a new transform:
        if transform_idx is None:
            transform_idx = len(transforms)
            transforms.append({"method": objective_type})
        transform_indices.append(transform_idx)

    ropt_config["objective_functions"] = {
        "names": names,
        "weights": weights,
        "scales": scales,
        "auto_scale": auto_scale,
    }
    if transforms:
        # Only needed if we specified at least one objective type:
        ropt_config["objective_functions"]["function_transforms"] = transform_indices
        ropt_config["function_transforms"] = transforms


def _parse_input_constraints(ever_config: EverestConfig, ropt_config, formatted_names):
    input_constrs = ever_config.input_constraints or None
    if input_constrs is None:
        return

    coefficients_matrix = []
    rhs_values = []
    types = []

    def _add_input_constraint(rhs_value, coefficients, constraint_type):
        if rhs_value is not None:
            coefficients_matrix.append(coefficients)
            rhs_values.append(rhs_value)
            types.append(constraint_type)

    for constr in input_constrs:
        coefficients = [0.0] * len(formatted_names)
        for name, value in constr.weights.items():
            coefficients[formatted_names.index(name)] = value
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


def _parse_output_constraints(ever_config: EverestConfig, ropt_config):
    ever_constrs = ever_config.output_constraints or None
    if ever_constrs is None:
        return

    names: List[str] = []
    rhs_values: List[float] = []
    scales: List[float] = []
    auto_scale: List[bool] = []
    types: List[ConstraintType] = []

    def _add_output_constraint(
        rhs_value: Optional[float], constraint_type: ConstraintType, suffix=None
    ):
        if rhs_value is not None:
            name = constr.name
            names.append(name if suffix is None else f"{name}:{suffix}")
            rhs_values.append(rhs_value)
            scales.append(constr.scale if constr.scale is not None else 1.0)
            auto_scale.append(constr.auto_scale or False)
            types.append(constraint_type)

    for constr in ever_constrs:
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
        "names": names,
        "rhs_values": rhs_values,
        "scales": scales,
        "auto_scale": auto_scale,
        "types": types,
    }


def _parse_optimization(ever_config: EverestConfig, ropt_config):
    ropt_config["optimizer"] = {}

    ever_opt = ever_config.optimization or None
    if ever_opt is None:
        return

    ropt_optimizer = ropt_config["optimizer"]
    ropt_gradient = ropt_config["gradient"]

    ropt_optimizer["backend"] = ever_opt.backend or "dakota"
    algorithm = ever_opt.algorithm or None
    if algorithm:
        ropt_optimizer["algorithm"] = algorithm

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
    options = ever_opt.options or []
    backend_options = ever_opt.backend_options or {}
    if options and backend_options:
        raise RuntimeError("Only one of 'options' and 'backend_options' allowed.")
    # The constraint_tolerance option is only used by Dakota:
    if ropt_optimizer["backend"] == "dakota":
        output_constraints = ever_config.output_constraints or None
        alg_const_tol = ever_opt.constraint_tolerance or None
        if output_constraints is not None and alg_const_tol is not None:
            options = options + [f"constraint_tolerance = {alg_const_tol}"]
    if options:
        ropt_optimizer["options"] = options
    if backend_options:
        ropt_optimizer["options"] = backend_options

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
            cvar_config: Dict[str, Any] = {
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
            objective_count = len(ropt_config["objective_functions"]["names"])
            constraint_count = len(
                ropt_config.get("nonlinear_constraints", {}).get("names", [])
            )
            ropt_config["objective_functions"]["realization_filters"] = (
                objective_count * [0]
            )
            if constraint_count > 0:
                ropt_config["nonlinear_constraints"]["realization_filters"] = (
                    constraint_count * [0]
                )
            cvar_config["options"].update({"sort": list(range(objective_count))})
            ropt_config["realization_filters"] = [cvar_config]
            # For efficiency, function and gradient evaluations should be split
            # so that no unnecessary gradients are calculated:
            ropt_optimizer["split_evaluations"] = True


def _parse_model(ever_config: EverestConfig, ropt_config):
    ever_model = ever_config.model or None
    if ever_model is None:
        return

    ever_reals = ever_model.realizations or []
    ever_reals_weights = ever_model.realizations_weights
    if ever_reals_weights is None:
        ever_reals_weights = [1.0 / len(ever_reals)] * len(ever_reals)

    ropt_config["realizations"] = {
        "names": ever_reals,
        "weights": ever_reals_weights,
    }
    ever_opt = ever_config.optimization or None
    min_real_succ = ever_opt.min_realizations_success if ever_opt is not None else None
    if min_real_succ is not None:
        ropt_config["realizations"]["realization_min_success"] = min_real_succ


def _parse_environment(ever_config: EverestConfig, ropt_config):
    ropt_config["optimizer"]["output_dir"] = os.path.abspath(
        ever_config.optimization_output_dir
    )


def everest2ropt(ever_config: EverestConfig):
    """Generate a ropt configuration from an Everest one

    NOTE: This method is a work in progress. So far only the some of
    the values are actually extracted, all the others are set to some
    more or less reasonable default
    """
    ropt_config: Dict[str, Any] = {}

    _parse_controls(ever_config, ropt_config)

    control_names = [
        (
            f"{control_name[0]}.{control_name[1]}-{control_name[2]}"
            if len(control_name) > 2
            else f"{control_name[0]}.{control_name[1]}"
        )
        for control_name in ropt_config["variables"]["names"]
    ]

    _parse_objectives(ever_config, ropt_config)
    _parse_input_constraints(ever_config, ropt_config, control_names)
    _parse_output_constraints(ever_config, ropt_config)
    _parse_optimization(ever_config, ropt_config)
    _parse_model(ever_config, ropt_config)
    _parse_environment(ever_config, ropt_config)

    return ropt_config
