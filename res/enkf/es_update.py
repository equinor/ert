import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from res.enkf.enums import RealizationStateEnum
from res._lib.enkf_analysis import UpdateSnapshot
from res._lib import update, analysis_module
from ert.analysis import ies

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
    ministep_snapshots: Dict[str, UpdateSnapshot] = field(default_factory=dict)


def analysis_smoother_update(
    updatestep,
    total_ens_size,
    obs,
    shared_rng,
    analysis_config,
    ensemble_config,
    source_fs,
    target_fs,
):
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

    if len(iens_active_index) < min(
        analysis_config.minimum_required_realizations, total_ens_size
    ):
        raise ErtAnalysisError(
            f"There are {iens_active_index} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )
    if len(updatestep) > 1 and module.name() == "IES_ENKF":
        raise ErtAnalysisError(
            "Can not combine IES_ENKF modules with multi step updates"
        )

    update.copy_parameters(source_fs, target_fs, ensemble_config, ens_mask)

    # Looping over local analysis ministep
    for ministep in updatestep:

        update_data = update.make_update_data(
            source_fs,
            target_fs,
            obs,
            ensemble_config,
            analysis_config,
            ens_mask,
            ministep,
            shared_rng,
        )
        # pylint: disable=unsupported-assignment-operation
        smoother_snapshot.ministep_snapshots[
            ministep.name()
        ] = update_data.update_snapshot
        if update_data.has_observations:

            """
            The update for one local_dataset instance consists of two main chunks:

            1. The first chunk updates all the parameters which don't have row
                scaling attached. These parameters are serialized together to the A
                matrix and all the parameters are updated in one go.

            2. The second chunk is loop over all the parameters which have row
                scaling attached. These parameters are updated one at a time.
            """

            module_config = analysis_module.get_module_config(module)
            module_data = analysis_module.get_module_data(module)

            if update_data.A is not None:
                if module_config.iterable:
                    ies.init_update(module_data, ens_mask, update_data.obs_mask)
                    iteration_nr = module_data.inc_iteration_nr()

                    ies.update_A(
                        module_data,
                        update_data.get_A(),
                        update_data.S,
                        update_data.R,
                        update_data.E,
                        update_data.D,
                        ies_inversion=module_config.inversion,
                        truncation=module_config.get_truncation(),
                        step_length=module_config.get_steplength(iteration_nr),
                    )
                else:
                    X = ies.make_X(
                        update_data.S,
                        update_data.R,
                        update_data.E,
                        update_data.D,
                        update_data.A,
                        ies_inversion=module_config.inversion,
                        truncation=module_config.get_truncation(),
                    )
                    update_data.A = update_data.A @ X

            if update_data.A_with_rowscaling:
                update.run_analysis_update_with_rowscaling(
                    module_config, module_data, update_data
                )

            update.save_parameters(
                target_fs, ensemble_config, iens_active_index, ministep, update_data
            )

        else:
            raise ErtAnalysisError(
                f"No active observations/parameters for MINISTEP: {ministep.name()}."
            )
    _write_update_report(
        Path(analysis_config.get_log_path()) / "deprecated", smoother_snapshot
    )
    return smoother_snapshot


def _write_update_report(fname: Path, snapshot: SmootherSnapshot):
    for ministep_name, ministep in snapshot.ministep_snapshots.items():
        with open(fname, "w") as fout:
            fout.write("=" * 127 + "\n")
            fout.write("Report step...: deprecated\n")
            fout.write(f"Ministep......: {ministep_name:<13}\n")
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
                    ministep.obs_name,
                    ministep.obs_value,
                    ministep.obs_std,
                    ministep.obs_status,
                    ministep.response_mean,
                    ministep.response_std,
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
    def __init__(self, enkf_main):
        self.ert = enkf_main

    @property
    def _analysis_config(self):
        return self.ert.analysisConfig()

    def hasModule(self, name):
        """
        Will check if we have analysis module @name.
        """
        return self._analysis_config.hasModule(name)

    def getModule(self, name):
        if self.hasModule(name):
            self._analysis_config.getModule(name)
        else:
            raise KeyError(f"No such module:{name}")

    def setGlobalStdScaling(self, weight):
        self._analysis_config.setGlobalStdScaling(weight)

    def smootherUpdate(self, run_context):
        source_fs = run_context.get_sim_fs()
        target_fs = run_context.get_target_fs()

        local_config = self.ert.getLocalConfig()
        updatestep = local_config.getUpdatestep()

        analysis_config = self.ert.analysisConfig()

        total_ens_size = self.ert.getEnsembleSize()
        obs = self.ert.getObservations()
        shared_rng = self.ert.rng()
        ensemble_config = self.ert.ensembleConfig()

        update_snapshot = analysis_smoother_update(
            updatestep,
            total_ens_size,
            obs,
            shared_rng,
            analysis_config,
            ensemble_config,
            source_fs,
            target_fs,
        )
        self.ert.update_snapshots[run_context.get_id()] = update_snapshot
