from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field

from ert.base_model_context import use_runtime_plugins
from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    DesignMatrix,
    ErtConfig,
    ObservationSettings,
    ParameterConfig,
)
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    MANUAL_ENIF_UPDATE_MODE,
    MANUAL_UPDATE_MODE,
    TEST_RUN_MODE,
)
from ert.plugins import get_site_plugins
from ert.validation import ActiveRange
from everest.config import EverestConfig

from .ensemble_experiment import EnsembleExperiment, EnsembleExperimentConfig
from .ensemble_information_filter import (
    EnsembleInformationFilter,
    EnsembleInformationFilterConfig,
)
from .ensemble_smoother import EnsembleSmoother, EnsembleSmootherConfig
from .evaluate_ensemble import EvaluateEnsemble, EvaluateEnsembleConfig
from .everest_run_model import EverestRunModel
from .initial_ensemble_run_model import DictEncodedDataFrame
from .manual_update import ManualUpdate, ManualUpdateConfig
from .manual_update_enif import ManualUpdateEnIF, ManualUpdateEnIFConfig
from .multiple_data_assimilation import (
    MultipleDataAssimilation,
    MultipleDataAssimilationConfig,
)
from .run_model import RunModel
from .single_test_run import SingleTestRun, SingleTestRunConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.namespace import Namespace
    from ert.run_models.event import StatusEvents


RunModelConfigs = Annotated[
    MultipleDataAssimilationConfig
    | EnsembleSmootherConfig
    | EnsembleInformationFilterConfig
    | SingleTestRunConfig
    | EnsembleExperimentConfig
    | ManualUpdateConfig
    | EvaluateEnsembleConfig
    | EverestConfig,
    Field(discriminator="type"),
]

logger = logging.getLogger(__name__)


def create_model(
    config: ErtConfig,
    args: Namespace,
    status_queue: SimpleQueue[StatusEvents],
) -> RunModel:
    logger.info(
        "Initiating experiment",
        extra={
            "mode": args.mode,
            "ensemble_size": config.runpath_config.num_realizations,
        },
    )
    runmodel_config = build_run_model_config(config, args)
    return _instantiate_run_model(runmodel_config, status_queue)


def build_run_model_config(config: ErtConfig, args: Namespace):
    update_settings = config.analysis_config.observation_settings

    if args.mode == TEST_RUN_MODE:
        return _setup_single_test_run(config, args)
    if args.mode == ENSEMBLE_EXPERIMENT_MODE:
        return _setup_ensemble_experiment(config, args)
    if args.mode == EVALUATE_ENSEMBLE_MODE:
        return _setup_evaluate_ensemble(config, args)
    if args.mode == ENSEMBLE_SMOOTHER_MODE:
        return _setup_ensemble_smoother(config, args, update_settings)
    if args.mode == ENIF_MODE:
        return _setup_ensemble_information_filter(config, args, update_settings)
    if args.mode == ES_MDA_MODE:
        return _setup_multiple_data_assimilation(config, args, update_settings)
    if args.mode == MANUAL_UPDATE_MODE:
        return _setup_manual_update(config, args, update_settings)
    if args.mode == MANUAL_ENIF_UPDATE_MODE:
        return _setup_manual_update_enif(config, args, update_settings)
    raise NotImplementedError(f"Run type not supported {args.mode}")


def _instantiate_run_model(
    runmodel_config: RunModelConfigs,
    status_queue: SimpleQueue[StatusEvents],
) -> RunModel:
    """Instantiate a RunModel from a config object."""
    if isinstance(runmodel_config, EverestConfig):
        site_plugins = get_site_plugins()
        with use_runtime_plugins(site_plugins):
            return EverestRunModel.create(
                everest_config=runmodel_config,
                experiment_name=f"EnOpt@{datetime.now().astimezone().isoformat(timespec='seconds')}",
                target_ensemble="batch",
                status_queue=status_queue,
                runtime_plugins=site_plugins,
            )

    model_map: dict[str, type[RunModel]] = {
        "single_test_run": SingleTestRun,
        "ensemble_experiment": EnsembleExperiment,
        "evaluate_ensemble": EvaluateEnsemble,
        "ensemble_smoother": EnsembleSmoother,
        "ensemble_information_filter": EnsembleInformationFilter,
        "multiple_data_assimilation": MultipleDataAssimilation,
        "manual_update": ManualUpdate,
        "manual_update_enif": ManualUpdateEnIF,
    }
    model_cls = model_map[runmodel_config.type]
    return model_cls(**runmodel_config.model_dump(), status_queue=status_queue)


def _merge_parameters(
    design_matrix: DesignMatrix | None,
    parameter_configs: list[ParameterConfig],
    require_updateable_param: bool = False,
) -> tuple[list[ParameterConfig], DictEncodedDataFrame | None]:
    if design_matrix is None:
        return parameter_configs, None

    merged_parameter_configs = design_matrix.merge_with_existing_parameters(
        parameter_configs
    )

    if require_updateable_param and not any(p.update for p in merged_parameter_configs):
        raise ConfigValidationError(
            "No parameters to update as all parameters were set to update:false!"
        )

    return merged_parameter_configs, DictEncodedDataFrame.from_polars(
        design_matrix.design_matrix_df
    )


def _setup_single_test_run(
    config: ErtConfig,
    args: Namespace,
) -> SingleTestRunConfig:
    experiment_name = (
        "single-test-run" if args.experiment_name is None else args.experiment_name
    )
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    if not active_realizations[0]:
        raise ConfigValidationError(
            "Cannot run single test run when the first realization is inactive."
        )

    parameter_configs, design_matrix = _merge_parameters(
        design_matrix=config.analysis_config.design_matrix,
        parameter_configs=config.ensemble_config.parameter_configuration,
    )

    return SingleTestRunConfig(
        random_seed=config.random_seed,
        runpath_file=config.runpath_file,
        active_realizations=[True],
        target_ensemble=args.current_ensemble,
        minimum_required_realizations=1,
        experiment_name=experiment_name,
        design_matrix=design_matrix,
        parameter_configuration=parameter_configs,
        response_configuration=config.ensemble_config.response_configuration,
        derived_response_configuration=config.ensemble_config.derived_response_configuration,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        storage_path=config.ens_path,
        queue_config=config.queue_config.create_local_copy(),
        observations=config.observation_declarations,
        shape_registry=config.shape_registry,
    )


def validate_minimum_realizations(
    config: ErtConfig, active_realizations: list[bool]
) -> None:
    min_realizations_count = config.analysis_config.minimum_required_realizations
    active_realizations_count = int(np.sum(active_realizations))
    if active_realizations_count < min_realizations_count:
        config.analysis_config.minimum_required_realizations = active_realizations_count
        ConfigWarning.warn(
            "MIN_REALIZATIONS was set to the current number of active realizations "
            f"({active_realizations_count}) as it is lower than the MIN_REALIZATIONS "
            f"({min_realizations_count}) that was specified in the config file."
        )


def _setup_ensemble_experiment(
    config: ErtConfig,
    args: Namespace,
) -> EnsembleExperimentConfig:
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    validate_minimum_realizations(config, active_realizations)
    experiment_name = args.experiment_name
    assert experiment_name is not None

    parameter_configs, design_matrix = _merge_parameters(
        design_matrix=config.analysis_config.design_matrix,
        parameter_configs=config.ensemble_config.parameter_configuration,
    )

    return EnsembleExperimentConfig(
        random_seed=config.random_seed,
        runpath_file=config.runpath_file,
        active_realizations=active_realizations,
        target_ensemble=args.current_ensemble,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        experiment_name=experiment_name,
        design_matrix=design_matrix,
        parameter_configuration=parameter_configs,
        response_configuration=config.ensemble_config.response_configuration,
        derived_response_configuration=config.ensemble_config.derived_response_configuration,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        observations=config.observation_declarations,
        shape_registry=config.shape_registry,
    )


def _setup_evaluate_ensemble(
    config: ErtConfig,
    args: Namespace,
) -> EvaluateEnsembleConfig:
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    validate_minimum_realizations(config, active_realizations)
    return EvaluateEnsembleConfig(
        random_seed=config.random_seed,
        active_realizations=active_realizations,
        ensemble_id=args.ensemble_id,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        runpath_file=config.runpath_file,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        shape_registry=config.shape_registry,
    )


def _get_and_validate_active_realizations_list(
    args: Namespace, config: ErtConfig
) -> list[bool]:
    if hasattr(args, "realizations") and args.realizations is not None:
        intersected_realizations = np.array(
            ActiveRange(
                rangestring=args.realizations,
                length=len(config.active_realizations),
            ).mask
        ) & np.array(config.active_realizations)
        if np.any(intersected_realizations):
            return intersected_realizations.tolist()
        elif (
            config.analysis_config.design_matrix is not None
            and config.analysis_config.design_matrix.active_realizations is not None
        ):
            raise ConfigValidationError(
                "The specified realizations do not intersect "
                "with the active realizations in the design matrix "
                "and NUM_REALIZATIONS."
            )
        else:
            raise ConfigValidationError(
                "The specified realizations do not intersect with NUM_REALIZATIONS."
            )
    return config.active_realizations


def _setup_manual_update(
    config: ErtConfig,
    args: Namespace,
    update_settings: ObservationSettings,
) -> ManualUpdateConfig:
    active_realizations = _realizations(args, config.runpath_config.num_realizations)
    validate_minimum_realizations(config, active_realizations.tolist())

    return ManualUpdateConfig(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        ensemble_id=args.ensemble_id,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        target_ensemble=args.target_ensemble,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        analysis_settings=config.analysis_config.es_settings,
        update_settings=update_settings,
        runpath_file=config.runpath_file,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        ert_templates=config.ert_templates,
        shape_registry=config.shape_registry,
    )


def _setup_manual_update_enif(
    config: ErtConfig,
    args: Namespace,
    update_settings: ObservationSettings,
) -> ManualUpdateEnIFConfig:
    active_realizations = _realizations(args, config.runpath_config.num_realizations)

    return ManualUpdateEnIFConfig(
        random_seed=config.random_seed,
        active_realizations=active_realizations.tolist(),
        ensemble_id=args.ensemble_id,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        target_ensemble=args.target_ensemble,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        analysis_settings=config.analysis_config.es_settings,
        update_settings=update_settings,
        runpath_file=config.runpath_file,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        shape_registry=config.shape_registry,
    )


def _setup_ensemble_smoother(
    config: ErtConfig,
    args: Namespace,
    update_settings: ObservationSettings,
) -> EnsembleSmootherConfig:
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    validate_minimum_realizations(config, active_realizations)
    if sum(active_realizations) < 2:
        raise ConfigValidationError(
            "Number of active realizations must be at least 2 for an update step"
        )

    parameter_configs, design_matrix = _merge_parameters(
        design_matrix=config.analysis_config.design_matrix,
        parameter_configs=config.ensemble_config.parameter_configuration,
        require_updateable_param=True,
    )

    return EnsembleSmootherConfig(
        target_ensemble=args.target_ensemble,
        experiment_name=getattr(args, "experiment_name", ""),
        active_realizations=active_realizations,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        random_seed=config.random_seed,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        analysis_settings=config.analysis_config.es_settings,
        update_settings=update_settings,
        runpath_file=config.runpath_file,
        design_matrix=design_matrix,
        parameter_configuration=parameter_configs,
        response_configuration=config.ensemble_config.response_configuration,
        derived_response_configuration=config.ensemble_config.derived_response_configuration,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        observations=config.observation_declarations,
        shape_registry=config.shape_registry,
    )


def _setup_ensemble_information_filter(
    config: ErtConfig,
    args: Namespace,
    update_settings: ObservationSettings,
) -> EnsembleInformationFilterConfig:
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    validate_minimum_realizations(config, active_realizations)
    if sum(active_realizations) < 2:
        raise ConfigValidationError(
            "Number of active realizations must be at least 2 for an update step"
        )

    parameter_configs, design_matrix = _merge_parameters(
        design_matrix=config.analysis_config.design_matrix,
        parameter_configs=config.ensemble_config.parameter_configuration,
    )

    return EnsembleInformationFilterConfig(
        target_ensemble=args.target_ensemble,
        experiment_name=getattr(args, "experiment_name", ""),
        active_realizations=active_realizations,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        random_seed=config.random_seed,
        storage_path=config.ens_path,
        queue_config=config.queue_config,
        analysis_settings=config.analysis_config.es_settings,
        update_settings=update_settings,
        runpath_file=config.runpath_file,
        design_matrix=design_matrix,
        parameter_configuration=parameter_configs,
        response_configuration=config.ensemble_config.response_configuration,
        derived_response_configuration=config.ensemble_config.derived_response_configuration,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        observations=config.observation_declarations,
        shape_registry=config.shape_registry,
    )


def _determine_restart_info(args: Namespace) -> tuple[bool, str | None]:
    """Handles differences in configuration between CLI and GUI.

    Returns
    -------
    A tuple containing the restart_run flag and the ensemble
    to run from.
    """
    if hasattr(args, "restart_ensemble_id"):
        # When running from CLI
        restart_run = args.restart_ensemble_id is not None
        prior_ensemble = args.restart_ensemble_id or ""
    else:
        # When running from GUI
        restart_run = args.restart_run
        prior_ensemble = args.prior_ensemble_id
    return restart_run, prior_ensemble


def _setup_multiple_data_assimilation(
    config: ErtConfig,
    args: Namespace,
    update_settings: ObservationSettings,
) -> MultipleDataAssimilationConfig:
    restart_run, prior_ensemble = _determine_restart_info(args)
    active_realizations = _get_and_validate_active_realizations_list(args, config)
    validate_minimum_realizations(config, active_realizations)
    if sum(active_realizations) < 2:
        raise ConfigValidationError(
            "Number of active realizations must be at least 2 for an update step"
        )

    parameter_configs, design_matrix = _merge_parameters(
        design_matrix=None if restart_run else config.analysis_config.design_matrix,
        parameter_configs=config.ensemble_config.parameter_configuration,
        require_updateable_param=True,
    )

    return MultipleDataAssimilationConfig(
        random_seed=config.random_seed,
        active_realizations=active_realizations,
        target_ensemble=_iterative_ensemble_format(args),
        weights=args.weights,
        restart_run=restart_run,
        prior_ensemble_id=prior_ensemble,
        minimum_required_realizations=config.analysis_config.minimum_required_realizations,
        experiment_name=args.experiment_name,
        queue_config=config.queue_config,
        update_settings=update_settings,
        storage_path=config.ens_path,
        analysis_settings=config.analysis_config.es_settings,
        runpath_file=config.runpath_file,
        design_matrix=design_matrix,
        parameter_configuration=parameter_configs,
        response_configuration=config.ensemble_config.response_configuration,
        derived_response_configuration=config.ensemble_config.derived_response_configuration,
        ert_templates=config.ert_templates,
        user_config_file=Path(config.user_config_file),
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        runpath_config=config.runpath_config,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        hooked_workflows=config.hooked_workflows,
        log_path=config.analysis_config.log_path,
        observations=config.observation_declarations,
        shape_registry=config.shape_registry,
    )


def _realizations(
    args: Namespace, ensemble_size_from_config: int
) -> npt.NDArray[np.bool_]:
    ensemble_size = (
        args.ensemble_size
        if hasattr(args, "ensemble_size") and args.ensemble_size is not None
        else ensemble_size_from_config
    )
    if not hasattr(args, "realizations") or args.realizations is None:
        return np.ones(ensemble_size, dtype=bool)
    return np.array(
        ActiveRange(rangestring=args.realizations, length=ensemble_size).mask
    )


def _iterative_ensemble_format(args: Namespace) -> str:
    """
    When a RunModel runs multiple iterations, an ensemble format will be used.
    E.g. when starting from the ensemble 'ensemble', subsequent runs can be named
    'ensemble_0', 'ensemble_1', 'ensemble_2', etc.

    This format can be set from the commandline via the `target_ensemble` option.
    or we use the current ensemble and add `_%d` to it.
    """
    return (
        args.target_ensemble
        or f"{getattr(args, 'current_ensemble', None) or 'default'}_%d"
    )
