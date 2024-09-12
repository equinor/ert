import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ropt.enums import ConstraintType, PerturbationType, VariableType
from typing_extensions import Final, TypeAlias

from everest.config import (
    ControlConfig,
    EverestConfig,
    SamplerConfig,
)
from everest.config.control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)

VariableName: TypeAlias = Tuple[str, str, int]
ControlName: TypeAlias = Union[Tuple[str, str], VariableName, List[VariableName]]
StrListDict: TypeAlias = DefaultDict[str, list]
IGNORE_KEYS: Final[Tuple[str, ...]] = (
    "enabled",
    "scaled_range",
    "auto_scale",
    "index",
    "name",
    "perturbation_magnitudes",
)


def _collect_sampler(
    sampler: Optional[SamplerConfig],
    storage: Dict[str, Any],
    control_name: Union[List[ControlName], ControlName, None] = None,
) -> Optional[Dict[str, Any]]:
    if sampler is None:
        return None
    map = sampler.model_dump(exclude_none=True, exclude={"backend", "method"})
    map["method"] = sampler.ropt_method
    control_names = map.setdefault("control_names", [])
    if control_name:
        control_names.extend(
            control_name if isinstance(control_name, list) else [control_name]
        )
    storage.setdefault("samplers", []).append(map)
    return map


def _scale_translations(
    is_scale: bool,
    _min: float,
    _max: float,
    lower_bound: float,
    upper_bound: float,
    perturbation_type: PerturbationType,
) -> Tuple[float, float, int]:
    if not is_scale:
        return 1.0, 0.0, perturbation_type.value
    scale = (_max - _min) / (upper_bound - lower_bound)
    return scale, _min - lower_bound * scale, PerturbationType.SCALED.value


@dataclass
class Control:
    name: Tuple[str, str]
    enabled: bool
    lower_bounds: float
    upper_bounds: float
    perturbation_magnitudes: Optional[float]
    initial_values: List[float]
    types: VariableType
    scaled_range: Tuple[float, float]
    auto_scale: bool
    index: Optional[int]
    scales: float
    offsets: float
    perturbation_types: int


def _resolve_everest_control(
    variable: Union[ControlVariableConfig, ControlVariableGuessListConfig],
    group: ControlConfig,
) -> Control:
    scaled_range = variable.scaled_range or group.scaled_range or (0, 1.0)
    auto_scale = variable.auto_scale or group.auto_scale
    lower_bound = group.min if variable.min is None else variable.min
    upper_bound = group.max if variable.max is None else variable.max

    scale, offset, perturbation_type = _scale_translations(
        auto_scale,
        lower_bound,  # type: ignore
        upper_bound,  # type: ignore
        *scaled_range,
        group.ropt_perturbation_type,
    )
    return Control(
        name=(group.name, variable.name),
        enabled=group.enabled if variable.enabled is None else variable.enabled,  # type: ignore
        lower_bounds=lower_bound,  # type: ignore
        upper_bounds=upper_bound,  # type: ignore
        perturbation_magnitudes=group.perturbation_magnitude
        if variable.perturbation_magnitude is None
        else variable.perturbation_magnitude,
        initial_values=group.initial_guess
        if variable.initial_guess is None
        else variable.initial_guess,  # type: ignore
        types=group.ropt_control_type
        if variable.ropt_control_type is None
        else variable.ropt_control_type,
        scaled_range=scaled_range,
        auto_scale=auto_scale,
        index=getattr(variable, "index", None),
        scales=scale,
        offsets=offset,
        perturbation_types=perturbation_type,
    )


def _variable_initial_guess_list_injection(
    control: Control,
    *,
    variables: StrListDict,
    gradients: StrListDict,
) -> List[VariableName]:
    guesses = len(control.initial_values)
    ropt_names = [(*control.name, index + 1) for index in range(guesses)]
    variables["names"].extend(ropt_names)
    variables["initial_values"].extend(control.initial_values)
    for key, value in asdict(control).items():
        if key not in (*IGNORE_KEYS, "initial_values"):
            (gradients if "perturbation" in key else variables)[key].extend(
                [value] * guesses
            )
    gradients["perturbation_magnitudes"].extend(
        [
            (
                (max(variables["upper_bounds"]) - min(variables["lower_bounds"])) / 10.0
                if control.perturbation_magnitudes is None
                else control.perturbation_magnitudes
            )
        ]
        * guesses
    )
    return ropt_names


def _variable_initial_guess_injection(
    control: Control,
    *,
    variables: StrListDict,
    gradients: StrListDict,
) -> ControlName:
    ropt_names: ControlName = (
        control.name if control.index is None else (*control.name, control.index)
    )
    variables["names"].append(ropt_names)
    for key, value in asdict(control).items():
        if key not in IGNORE_KEYS:
            (gradients if "perturbation" in key else variables)[key].append(value)
    gradients["perturbation_magnitudes"].append(
        (max(variables["upper_bounds"]) - min(variables["lower_bounds"])) / 10.0
        if control.perturbation_magnitudes is None
        else control.perturbation_magnitudes
    )
    return ropt_names


def _parse_controls(controls: Sequence[ControlConfig], ropt_config):
    """Extract info from ever_config['controls']"""
    enabled = []
    variables: StrListDict = defaultdict(list)
    gradients: StrListDict = defaultdict(list)

    for group in controls:
        sampler = _collect_sampler(group.sampler, ropt_config)

        for variable in group.variables:
            control = _resolve_everest_control(variable, group)
            enabled.append(control.enabled)
            control_injector = (
                _variable_initial_guess_list_injection
                if isinstance(variable.initial_guess, list)
                else _variable_initial_guess_injection
            )
            ropt_names = control_injector(
                control,
                variables=variables,
                gradients=gradients,
            )

            if (
                _collect_sampler(variable.sampler, ropt_config, ropt_names) is None
                and sampler
            ):
                control_names = sampler["control_names"]
                (
                    control_names.extend
                    if isinstance(ropt_names, list)
                    else control_names.append
                )(ropt_names)

    ropt_config["variables"] = dict(variables)
    ropt_config["variables"]["indices"] = (
        None if all(enabled) else [idx for idx, item in enumerate(enabled) if item]
    )
    ropt_config["variables"]["delimiters"] = ".-"
    ropt_config["gradient"] = dict(gradients)

    # The samplers in the list constructed above contain the names of the
    # variables they should apply to, but ropt expects a array of indices that
    # map variables to the samplers that should apply to them:
    if samplers := ropt_config.get("samplers"):
        sampler_indices = [0] * len(variables["names"])
        for idx, sampler in enumerate(samplers):
            for name in sampler.pop("control_names"):  # type: ignore
                sampler_indices[variables["names"].index(name)] = idx
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

    backend = ever_opt.backend or "dakota"
    algorithm = ever_opt.algorithm or "default"
    ropt_optimizer["method"] = f"{backend}/{algorithm}"

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
    if backend == "dakota":
        output_constraints = ever_config.output_constraints or None
        alg_const_tol = ever_opt.constraint_tolerance or None
        if output_constraints is not None and alg_const_tol is not None:
            options += [f"constraint_tolerance = {alg_const_tol}"]
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

    _parse_controls(ever_config.controls, ropt_config)

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
