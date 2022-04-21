import logging
from typing_extensions import Literal
from res import _lib
from res.enkf.enums import RealizationStateEnum


logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


def size_is_big_enough(active_ens_size, required_ens_size):
    if active_ens_size < required_ens_size:
        logger.error(
            "** ERROR **: There are %d active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
            active_ens_size,
        )
        return False
    return True


def config_is_correct(updatestep, module_name: Literal["IES_ENKF", "STD_ENKF"]):
    # exit if multi step update with iterable modules
    if len(updatestep) > 1 and module_name == "IES_ENKF":
        logger.error(
            "** ERROR **: Can not combine IES_ENKF modules with multi step "
            "updates - sorry"
        )
        return False
    return True


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

    if not size_is_big_enough(
        len(iens_active_index),
        min(analysis_config.minimum_required_realizations, total_ens_size),
    ) or not config_is_correct(
        updatestep,
        module.name(),
    ):
        return False

    _lib.update.copy_parameters(source_fs, target_fs, ensemble_config, ens_mask)

    # Looping over local analysis ministep
    for i in range(len(updatestep)):
        ministep = updatestep[i]

        update_data = _lib.update.make_update_data(
            source_fs,
            target_fs,
            obs,
            ensemble_config,
            analysis_config,
            ens_mask,
            ministep,
            shared_rng,
        )
        if update_data.has_observations:

            """
            The update for one local_dataset instance consists of two main chunks:

            1. The first chunk updates all the parameters which don't have row
                scaling attached. These parameters are serialized together to the A
                matrix and all the parameters are updated in one go.

            2. The second chunk is loop over all the parameters which have row
                scaling attached. These parameters are updated one at a time.
            """

            module_config = _lib.analysis_module.get_module_config(module)
            module_data = _lib.analysis_module.get_module_data(module)
            if update_data.A is not None:
                _lib.update.run_analysis_update_without_rowscaling(
                    module_config, module_data, ens_mask, update_data
                )
            if update_data.A_with_rowscaling:
                _lib.update.run_analysis_update_with_rowscaling(
                    module_config, module_data, update_data
                )

            _lib.update.save_parameters(
                target_fs, ensemble_config, iens_active_index, ministep, update_data
            )

        else:
            logger.error(
                "No active observations/parameters for MINISTEP: %s.", ministep.name()
            )

    return True


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
            raise KeyError("No such module:%s " % name)

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

        return analysis_smoother_update(
            updatestep,
            total_ens_size,
            obs,
            shared_rng,
            analysis_config,
            ensemble_config,
            source_fs,
            target_fs,
        )
