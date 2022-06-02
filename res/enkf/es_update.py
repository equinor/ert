import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, Optional

from ecl.util.util import RandomNumberGenerator
from res.enkf.enums import RealizationStateEnum
from res.enkf.analysis_config import AnalysisConfig
from res.enkf.ensemble_config import EnsembleConfig
from res._lib.enkf_analysis import UpdateSnapshot
from res._lib import update, analysis_module
from res.analysis.configuration import UpdateConfiguration
from res.enkf.enkf_obs import EnkfObs
from res.enkf.enkf_fs import EnkfFs
from ert.analysis import ies

if TYPE_CHECKING:
    from res.enkf import EnKFMain, ErtRunContext

logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


@dataclass
class SmootherSnapshot:
    source_case: str
    target_case: str
    analyis_module: str
    analysis_configuration: Dict[str, Any]
    alpha: float
    std_cutoff: float
    update_step_snapshots: Dict[str, UpdateSnapshot] = field(default_factory=dict)


def analysis_smoother_update(
    updatestep: UpdateConfiguration,
    obs: EnkfObs,
    shared_rng: RandomNumberGenerator,
    analysis_config: AnalysisConfig,
    ensemble_config: EnsembleConfig,
    source_fs: EnkfFs,
    target_fs: EnkfFs,
    w_container: Optional[ies.ModuleData],
) -> SmootherSnapshot:
    source_state_map = source_fs.getStateMap()

    ens_mask = source_state_map.selectMatching(RealizationStateEnum.STATE_HAS_DATA)
    iens_active_index = [i for i in range(len(ens_mask)) if ens_mask[i]]
    module = analysis_config.getActiveModule()

    smoother_snapshot = SmootherSnapshot(
        source_fs.getCaseName(),
        target_fs.getCaseName(),
        analysis_config.activeModuleName(),
        {
            name: analysis_config.getActiveModule().getVariableValue(name)
            for name in analysis_config.getActiveModule().getVariableNames()
        },
        analysis_config.getEnkfAlpha(),
        analysis_config.getStdCutoff(),
    )

    if not analysis_config.haveEnoughRealisations(len(iens_active_index)):
        raise ErtAnalysisError(
            f"There are {len(iens_active_index)} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )
    if len(updatestep) > 1 and module.name() == "IES_ENKF":
        raise ErtAnalysisError(
            "Can not combine IES_ENKF modules with multi step updates"
        )

    update.copy_parameters(source_fs, target_fs, ensemble_config, ens_mask)
    alpha = analysis_config.getEnkfAlpha()
    std_cutoff = analysis_config.getStdCutoff()
    global_scaling = analysis_config.getGlobalStdScaling()

    # Looping over local analysis update_step
    for update_step in updatestep:

        S, observation_handle = update.load_observations_and_responses(
            source_fs,
            obs,
            alpha,
            std_cutoff,
            global_scaling,
            ens_mask,
            update_step.observation_config(),
        )
        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.update_step_snapshots[
            update_step.name
        ] = observation_handle.update_snapshot
        observation_values = observation_handle.observation_values
        observation_errors = observation_handle.observation_errors
        observation_mask = observation_handle.obs_mask
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        A = update.load_parameters(
            target_fs, ensemble_config, iens_active_index, update_step.parameters
        )
        A_with_rowscaling = update.load_row_scaling_parameters(
            target_fs,
            ensemble_config,
            iens_active_index,
            update_step.row_scaling_parameters,
        )
        noise = update.generate_noise(
            len(observation_values), len(iens_active_index), shared_rng
        )
        E = ies.make_E(observation_errors, noise)
        R = np.identity(len(observation_errors))
        D = ies.make_D(observation_values, E, S)
        D = (D.T / observation_errors).T
        E = (E.T / observation_errors).T
        S = (S.T / observation_errors).T
        module_config = analysis_module.get_module_config(module)

        if A is not None:
            if module_config.iterable:
                if w_container is None:
                    raise ErtAnalysisError(
                        "Trying to run IES without coffecient matrix container!"
                    )
                ies.init_update(w_container, ens_mask, observation_mask)

                ies.update_A(
                    w_container,
                    A,
                    S,
                    R,
                    E,
                    D,
                    ies_inversion=module_config.inversion,
                    truncation=module_config.get_truncation(),
                    step_length=module_config.get_steplength(w_container.iteration_nr),
                )
            else:
                X = ies.make_X(
                    S,
                    R,
                    E,
                    D,
                    A,
                    ies_inversion=module_config.inversion,
                    truncation=module_config.get_truncation(),
                )
                A = A @ X

            update.save_parameters(
                target_fs,
                ensemble_config,
                iens_active_index,
                update_step.parameters,
                A,
            )

        if A_with_rowscaling:
            if module_config.iterable:
                raise ErtAnalysisError(
                    "Sorry - row scaling for distance based "
                    "localization can not be combined with "
                    "analysis modules which update the A matrix"
                )
            for (A, row_scaling) in A_with_rowscaling:
                X = ies.make_X(
                    S,
                    R,
                    E,
                    D,
                    A,
                    ies_inversion=module_config.inversion,
                    truncation=module_config.get_truncation(),
                )
                row_scaling.multiply(A, X)

            update.save_row_scaling_parameters(
                target_fs,
                ensemble_config,
                iens_active_index,
                update_step.row_scaling_parameters,
                A_with_rowscaling,
            )

    _write_update_report(
        Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
    )
    return smoother_snapshot


def _write_update_report(fname: Path, snapshot: SmootherSnapshot) -> None:
    for update_step_name, update_step in snapshot.update_step_snapshots.items():
        with open(fname, "w") as fout:
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


class ESUpdate:
    def __init__(self, enkf_main: "EnKFMain"):
        self.ert = enkf_main

    def setGlobalStdScaling(self, weight: float) -> None:
        self.ert.analysisConfig().setGlobalStdScaling(weight)

    def smootherUpdate(
        self, run_context: "ErtRunContext", w_container: Optional[ies.ModuleData] = None
    ) -> None:
        source_fs = run_context.get_sim_fs()
        target_fs = run_context.get_target_fs()

        updatestep = self.ert.getLocalConfig()

        analysis_config = self.ert.analysisConfig()

        obs = self.ert.getObservations()
        shared_rng = self.ert.rng()
        ensemble_config = self.ert.ensembleConfig()

        update_snapshot = analysis_smoother_update(
            updatestep,
            obs,
            shared_rng,
            analysis_config,
            ensemble_config,
            source_fs,
            target_fs,
            w_container,
        )
        self.ert.update_snapshots[run_context.get_id()] = update_snapshot
