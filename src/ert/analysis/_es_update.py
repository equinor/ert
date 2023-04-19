from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import iterative_ensemble_smoother as ies
import numpy as np
import pandas
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)
from pandas import DataFrame

from ert._c_wrappers.enkf.config.field_config import Field
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig
from ert._c_wrappers.enkf.enums import ActiveMode, ErtImplType, RealizationStateEnum
from ert._c_wrappers.enkf.row_scaling import RowScaling
from ert._clib import update

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.analysis import AnalysisModule
    from ert._c_wrappers.analysis.configuration import UpdateConfiguration
    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs
    from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
    from ert.storage import EnsembleAccessor, EnsembleReader

logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


@dataclass
class UpdateSnapshot:
    obs_name: List[str]
    obs_value: List[float]
    obs_std: List[float]
    obs_status: List[str]
    response_mean: List[float]
    response_std: List[float]


@dataclass
class SmootherSnapshot:
    source_case: str
    target_case: str
    analysis_module: str
    analysis_configuration: Dict[str, Any]
    alpha: float
    std_cutoff: float
    update_step_snapshots: Dict[str, "UpdateSnapshot"] = field(default_factory=dict)


def _get_A_matrix(
    temporary_storage: Dict[str, "npt.NDArray[np.double]"],
    parameters: List[update.Parameter],
) -> Optional["npt.NDArray[np.double]"]:
    matrices: List["npt.NDArray[np.double]"] = []
    for p in parameters:
        if p.active_list.getMode() == ActiveMode.ALL_ACTIVE:
            matrices.append(temporary_storage[p.name])
        elif p.active_list.getMode() == ActiveMode.PARTLY_ACTIVE:
            matrices.append(
                temporary_storage[p.name][p.active_list.get_active_index_list(), :]
            )
    return np.vstack(matrices) if matrices else None


def _param_ensemble_for_projection(
    temp_storage: Dict[str, npt.NDArray[np.double]],
    ensemble_size: int,
    updatestep: UpdateConfiguration,
) -> Optional[npt.NDArray[np.double]]:
    """Responses must be projected when num_params < ensemble_size - 1.
    The number of parameters here refers to the total number of parameters used to
    drive the model and not the number of parameters in a local update step.

    Scenario 1:
        - p < N - 1, meaning that projection needs to be done.
        - User wants to loop through each parameter and update it separately,
          perhaps to do distance based localization.
        In this case, we need to pass `param_ensemble` to the smoother
        so that the responses are projected using the same `param_ensemble`
        in every loop.
    Scenario 2:
        - p > N - 1, meaning that projection is not necessary.
        - User wants to loop through every parameter as in Scenario 1.
        In this case `param_ensemble` should be `None` which means
        that no projection will be done even when updating a single parameter.
    """
    num_params = sum(arr.shape[0] for arr in temp_storage.values())
    if num_params < ensemble_size - 1:
        _params: List[List[update.Parameter]] = [
            update_step.parameters for update_step in updatestep
        ]
        # Flatten list of lists
        params: List[update.Parameter] = [
            item for sublist in _params for item in sublist
        ]
        return _get_A_matrix(temp_storage, params)
    return None


def _get_row_scaling_A_matrices(
    temporary_storage: Dict[str, "npt.NDArray[np.double]"],
    parameters: List[update.RowScalingParameter],
) -> List[Tuple["npt.NDArray[np.double]", RowScaling]]:
    matrices = []
    for p in parameters:
        if p.active_list.getMode() == ActiveMode.ALL_ACTIVE:
            matrices.append((temporary_storage[p.name], p.row_scaling))
        elif p.active_list.getMode() == ActiveMode.PARTLY_ACTIVE:
            matrices.append(
                (
                    temporary_storage[p.name][p.active_list.get_active_index_list(), :],
                    p.row_scaling,
                ),
            )

    return matrices


def _save_to_temporary_storage(
    temporary_storage: Dict[str, "npt.NDArray[np.double]"],
    parameters: List[update.Parameter],
    A: Optional["npt.NDArray[np.double]"],
) -> None:
    if A is None:
        return
    offset = 0
    for p in parameters:
        if p.active_list.getMode() == ActiveMode.ALL_ACTIVE:
            rows = temporary_storage[p.name].shape[0]
            temporary_storage[p.name] = A[offset : offset + rows, :]
            offset += rows
        elif p.active_list.getMode() == ActiveMode.PARTLY_ACTIVE:
            row_indices = p.active_list.get_active_index_list()
            for i, row in enumerate(row_indices):
                temporary_storage[p.name][row] = A[offset + i]
            offset += len(row_indices)


def _save_temporary_storage_to_disk(
    target_fs: EnsembleAccessor,
    ensemble_config: "EnsembleConfig",
    temporary_storage: Dict[str, "npt.NDArray[np.double]"],
    iens_active_index: List[int],
) -> None:
    for key, matrix in temporary_storage.items():
        config_node = ensemble_config.getNode(key)
        if config_node.getImplementationType() == ErtImplType.GEN_KW:
            gen_kw_config = config_node.getKeywordModelConfig()
            parameter_keys = list(gen_kw_config)
            target_fs.save_gen_kw(
                key,
                parameter_keys,
                iens_active_index,
                matrix,
            )
        elif isinstance(config_node, SurfaceConfig):
            for i, realization in enumerate(iens_active_index):
                target_fs.save_surface_data(key, realization, matrix[:, i])
        elif isinstance(config_node, Field):
            for i, realization in enumerate(iens_active_index):
                target_fs.save_field(key, realization, matrix[:, i], unmasked=True)
        else:
            raise NotImplementedError(
                f"{config_node.getImplementationType()} is not supported"
            )


def _create_temporary_parameter_storage(
    source_fs: EnsembleReader,
    ensemble_config: "EnsembleConfig",
    iens_active_index: List[int],
) -> Dict[str, "npt.NDArray[np.double]"]:
    temporary_storage = {}
    for key in ensemble_config.parameters:
        config_node = ensemble_config.getNode(key)
        if config_node.getImplementationType() == ErtImplType.GEN_KW:
            matrix = source_fs.load_gen_kw(key, iens_active_index)
        elif isinstance(config_node, SurfaceConfig):
            matrix = source_fs.load_surface_data(key, iens_active_index)
        elif isinstance(config_node, Field):
            matrix = source_fs.load_field(key, iens_active_index)
        else:
            raise NotImplementedError(
                f"{config_node.getImplementationType()} is not supported"
            )
        temporary_storage[key] = matrix
    return temporary_storage


def _get_obs_and_measure_data(
    obs: "EnkfObs",
    source_fs: EnsembleReader,
    selected_observations: List[Tuple[str, List[int]]],
    ens_active_list: List[int],
) -> Tuple[DataFrame, DataFrame]:
    data_keys = defaultdict(set)
    obs_data = []
    for obs_key, active_list in selected_observations:
        obs_vector = obs[obs_key]

        try:
            data_key = obs_vector.getDataKey()
        except KeyError:
            raise KeyError(f"No data key for obs key: {obs_key}")
        imp_type = obs_vector.getImplementationType().name
        if imp_type == "GEN_OBS":
            obs_data.append(obs_vector.get_gen_obs_data(active_list))
            data_key = f"{data_key}@{obs_vector.activeStep()}"
        elif imp_type == "SUMMARY_OBS":
            obs_data.append(obs_vector.get_summary_obs_data(obs, active_list))

        data_keys[imp_type].add(data_key)

    measured_data = []
    for imp_type, keys in data_keys.items():
        if imp_type == "SUMMARY_OBS":
            measured_data.append(
                source_fs.load_summary_data_as_df(list(keys), ens_active_list)
            )

        if imp_type == "GEN_OBS":
            measured_data.append(
                source_fs.load_gen_data_as_df(list(keys), ens_active_list)
            )

    return pandas.concat(measured_data), pandas.concat(obs_data)


def _deactivate_outliers(
    meas_data: DataFrame, std_cutoff: float, alpha: float, global_std_scaling: float
) -> pandas.Index:
    """
    Extracts indices for which outliers that are to be extracted
    """
    filter_std = _filter_ensemble_std(meas_data, std_cutoff)
    filter_mean_obs = _filter_ensemble_mean_obs(meas_data, alpha, global_std_scaling)
    return filter_std.index.union(filter_mean_obs.index)


def _filter_ensemble_std(data: DataFrame, std_cutoff: float) -> pandas.Series:
    """
    Filters on ensemble variation versus a user defined standard
    deviation cutoff. If there is not enough variation in the measurements
    the data point is removed.
    """
    S = data.loc[:, ~data.columns.isin(["OBS", "STD"])]
    ens_std = S.std(axis=1, ddof=0)
    std_filter = ens_std <= std_cutoff
    return std_filter[std_filter]


def _filter_ensemble_mean_obs(
    data: DataFrame, alpha: float, global_std_scaling: float
) -> pandas.Series:
    """
    Filters on distance between the observed data and the ensemble mean
    based on variation and a user defined alpha.
    """
    S = data.loc[:, ~data.columns.isin(["OBS", "STD"])]
    ens_mean = S.mean(axis=1)
    ens_std = S.std(axis=1, ddof=0)
    obs_values = data.loc[:, "OBS"]
    obs_std = data.loc[:, "STD"] * global_std_scaling

    mean_filter = abs(obs_values - ens_mean) > alpha * (ens_std + obs_std)
    return mean_filter[mean_filter]


def _create_update_snapshot(data: DataFrame, obs_mask: List[bool]) -> UpdateSnapshot:
    observation_values = data.loc[:, "OBS"].to_numpy()
    observation_errors = data.loc[:, "STD"].to_numpy()
    S = data.loc[:, ~data.columns.isin(["OBS", "STD"])]

    return UpdateSnapshot(
        obs_name=[obs_key for (obs_key, _, _) in data.index.to_list()],
        obs_value=observation_values,
        obs_std=observation_errors,
        obs_status=["ACTIVE" if v else "DEACTIVATED" for v in obs_mask],
        response_mean=S.mean(axis=1),
        response_std=S.std(axis=1, ddof=0),
    )


def _load_observations_and_responses(
    source_fs: EnsembleReader,
    obs: "EnkfObs",
    alpha: float,
    std_cutoff: float,
    global_std_scaling: float,
    ens_mask: List[bool],
    selected_observations: List[Tuple[str, List[int]]],
) -> Any:
    ens_active_list = [i for i, b in enumerate(ens_mask) if b]
    measured_data, obs_data = _get_obs_and_measure_data(
        obs, source_fs, selected_observations, ens_active_list
    )

    joined = obs_data.join(measured_data, on=["data_key", "axis"], how="inner")

    if joined.isna().any().any():
        raise ErtAnalysisError("Missing response for observations")
    if len(obs_data) > len(joined):
        missing_indices = set(obs_data.index) - set(joined.index)
        error_msg = []
        for i in missing_indices:
            error_msg.append(
                f"Observation: {i[0]} attached to response: {i[1]} "
                f"at: {i[2]} has no response"
            )
        raise IndexError("\n".join(error_msg))

    obs_filter = _deactivate_outliers(joined, std_cutoff, alpha, global_std_scaling)
    obs_mask = [i not in obs_filter for i in joined.index]

    # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    # in for example evensen2018 - Analysis of iterative ensemble smoothers for
    # solving inverse problems.
    # `global_std_scaling` is 1.0 for ES.
    joined.loc[:, "STD"] *= sqrt(global_std_scaling)
    joined.drop(index=obs_filter, inplace=True)
    update_snapshot = _create_update_snapshot(joined, obs_mask)

    observation_values = joined.loc[:, "OBS"].to_numpy()
    observation_errors = joined.loc[:, "STD"].to_numpy()
    S = joined.loc[:, ~joined.columns.isin(["OBS", "STD"])]

    return S.to_numpy(), (
        observation_values,
        observation_errors,
        obs_mask,
        update_snapshot,
    )


def analysis_ES(
    updatestep: "UpdateConfiguration",
    obs: "EnkfObs",
    rng: np.random.Generator,
    module: "AnalysisModule",
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: List[bool],
    ensemble_config: "EnsembleConfig",
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
) -> None:
    iens_active_index = [i for i in range(len(ens_mask)) if ens_mask[i]]

    temp_storage = _create_temporary_parameter_storage(
        source_fs, ensemble_config, iens_active_index
    )

    ensemble_size = sum(ens_mask)
    param_ensemble = _param_ensemble_for_projection(
        temp_storage, ensemble_size, updatestep
    )

    # Looping over local analysis update_step
    for update_step in updatestep:
        try:
            S, (
                observation_values,
                observation_errors,
                _,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                obs,
                alpha,
                std_cutoff,
                global_scaling,
                ens_mask,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(e) from e

        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        A_with_rowscaling = _get_row_scaling_A_matrices(
            temp_storage, update_step.row_scaling_parameters
        )
        noise = rng.standard_normal(size=(len(observation_values), S.shape[1]))

        smoother = ies.ES()
        for parameter in update_step.parameters:
            smoother.fit(
                S,
                observation_errors,
                observation_values,
                noise=noise,
                truncation=module.get_truncation(),
                inversion=ies.InversionType(module.inversion),
                param_ensemble=param_ensemble,
            )
            if active_indices := parameter.active_list.get_active_index_list():
                temp_storage[parameter.name][active_indices, :] = smoother.update(
                    temp_storage[parameter.name][active_indices, :]
                )
            else:
                temp_storage[parameter.name] = smoother.update(
                    temp_storage[parameter.name]
                )

        if A_with_rowscaling:
            A_with_rowscaling = ensemble_smoother_update_step_row_scaling(
                S,
                A_with_rowscaling,
                observation_errors,
                observation_values,
                noise,
                module.get_truncation(),
                ies.InversionType(module.inversion),
            )
            for parameter, (A, _) in zip(
                update_step.row_scaling_parameters, A_with_rowscaling
            ):
                _save_to_temporary_storage(temp_storage, [parameter], A)

    _save_temporary_storage_to_disk(
        target_fs, ensemble_config, temp_storage, iens_active_index
    )


def analysis_IES(
    updatestep: "UpdateConfiguration",
    obs: "EnkfObs",
    rng: np.random.Generator,
    module: "AnalysisModule",
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: List[bool],
    ensemble_config: "EnsembleConfig",
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    iterative_ensemble_smoother: ies.SIES,
) -> None:
    iens_active_index = [i for i in range(len(ens_mask)) if ens_mask[i]]

    temp_storage = _create_temporary_parameter_storage(
        source_fs, ensemble_config, iens_active_index
    )

    ensemble_size = sum(ens_mask)
    param_ensemble = _param_ensemble_for_projection(
        temp_storage, ensemble_size, updatestep
    )

    # Looping over local analysis update_step
    for update_step in updatestep:
        try:
            S, (
                observation_values,
                observation_errors,
                observation_mask,
                update_snapshot,
            ) = _load_observations_and_responses(
                source_fs,
                obs,
                alpha,
                std_cutoff,
                global_scaling,
                ens_mask,
                update_step.observation_config(),
            )
        except IndexError as e:
            raise ErtAnalysisError(e)
        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        noise = rng.standard_normal(size=(len(observation_values), S.shape[1]))
        for parameter in update_step.parameters:
            iterative_ensemble_smoother.fit(
                S,
                observation_errors,
                observation_values,
                noise=noise,
                ensemble_mask=np.array(ens_mask),
                inversion=ies.InversionType(module.inversion),
                truncation=module.get_truncation(),
                param_ensemble=param_ensemble,
            )
            if active_indices := parameter.active_list.get_active_index_list():
                temp_storage[parameter.name][
                    active_indices, :
                ] = iterative_ensemble_smoother.update(
                    temp_storage[parameter.name][active_indices, :]
                )
            else:
                temp_storage[parameter.name] = iterative_ensemble_smoother.update(
                    temp_storage[parameter.name]
                )

    _save_temporary_storage_to_disk(
        target_fs, ensemble_config, temp_storage, iens_active_index
    )


def _write_update_report(fname: Path, snapshot: SmootherSnapshot) -> None:
    # Make sure log file parents exist
    fname.parent.mkdir(parents=True, exist_ok=True)
    for update_step_name, update_step in snapshot.update_step_snapshots.items():
        with open(fname, "w", encoding="utf-8") as fout:
            fout.write("=" * 127 + "\n")
            fout.write("Report step...: deprecated\n")
            fout.write(f"Update step......: {update_step_name:<10}\n")
            fout.write("-" * 127 + "\n")
            fout.write(
                "Observed history".rjust(73)
                + "|".rjust(16)
                + "Simulated data".rjust(27)
                + "\n".rjust(9)
            )
            fout.write("-" * 127 + "\n")
            for nr, (name, val, std, status, ens_val, ens_std) in enumerate(
                zip(
                    update_step.obs_name,
                    update_step.obs_value,
                    update_step.obs_std,
                    update_step.obs_status,
                    update_step.response_mean,
                    update_step.response_std,
                )
            ):
                if status in ["DEACTIVATED", "LOCAL_INACTIVE"]:
                    status = "Inactive"
                fout.write(
                    f"{nr+1:^6}: {name:30} {val:>16.3f} +/- {std:>17.3f} "
                    f"{status.capitalize():9} | {ens_val:>17.3f} +/- {ens_std:>15.3f}  "
                    f"\n"
                )
            fout.write("=" * 127 + "\n")


def _assert_has_enough_realizations(
    ens_mask: List[bool], analysis_config: "AnalysisConfig"
) -> None:
    active_realizations = sum(ens_mask)
    if not analysis_config.have_enough_realisations(active_realizations):
        raise ErtAnalysisError(
            f"There are {active_realizations} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )


def _create_smoother_snapshot(
    prior_name: "str", posterior_name: "str", analysis_config: "AnalysisConfig"
) -> SmootherSnapshot:
    return SmootherSnapshot(
        prior_name,
        posterior_name,
        analysis_config.active_module_name(),
        analysis_config.get_active_module().variable_value_dict(),
        analysis_config.get_enkf_alpha(),
        analysis_config.get_std_cutoff(),
    )


class ESUpdate:
    def __init__(self, enkf_main: "EnKFMain"):
        self.ert = enkf_main
        self.update_snapshots: Dict[str, SmootherSnapshot] = {}

    def smootherUpdate(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        run_id: str,
    ) -> None:
        updatestep = self.ert.getLocalConfig()

        analysis_config = self.ert.analysisConfig()

        obs = self.ert.getObservations()
        ensemble_config = self.ert.ensembleConfig()

        alpha = analysis_config.get_enkf_alpha()
        std_cutoff = analysis_config.get_std_cutoff()
        global_scaling = analysis_config.get_global_std_scaling()
        ens_mask = prior_storage.get_realization_mask_from_state(
            [RealizationStateEnum.STATE_HAS_DATA]
        )
        _assert_has_enough_realizations(ens_mask, analysis_config)

        smoother_snapshot = _create_smoother_snapshot(
            prior_storage.name, posterior_storage.name, analysis_config
        )

        analysis_ES(
            updatestep,
            obs,
            self.ert.rng(),
            analysis_config.get_active_module(),
            alpha,
            std_cutoff,
            global_scaling,
            smoother_snapshot,
            ens_mask,
            ensemble_config,
            prior_storage,
            posterior_storage,
        )

        _write_update_report(
            Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
        )

        self.update_snapshots[run_id] = smoother_snapshot

    def iterative_smoother_update(
        self,
        prior_storage: EnsembleReader,
        posterior_storage: EnsembleAccessor,
        w_container: ies.SIES,
        run_id: str,
    ) -> None:
        updatestep = self.ert.getLocalConfig()
        if len(updatestep) > 1:
            raise ErtAnalysisError(
                "Can not combine IES_ENKF modules with multi step updates"
            )

        analysis_config = self.ert.analysisConfig()

        obs = self.ert.getObservations()
        ensemble_config = self.ert.ensembleConfig()

        alpha = analysis_config.get_enkf_alpha()
        std_cutoff = analysis_config.get_std_cutoff()
        global_scaling = analysis_config.get_global_std_scaling()
        ens_mask = prior_storage.get_realization_mask_from_state(
            [RealizationStateEnum.STATE_HAS_DATA]
        )

        _assert_has_enough_realizations(ens_mask, analysis_config)

        smoother_snapshot = _create_smoother_snapshot(
            prior_storage.name, posterior_storage.name, analysis_config
        )

        analysis_IES(
            updatestep,
            obs,
            self.ert.rng(),
            analysis_config.get_active_module(),
            alpha,
            std_cutoff,
            global_scaling,
            smoother_snapshot,
            ens_mask,
            ensemble_config,
            prior_storage,
            posterior_storage,
            w_container,
        )

        _write_update_report(
            Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
        )

        self.update_snapshots[run_id] = smoother_snapshot
