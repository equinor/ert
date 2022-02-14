from res import _lib


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
        verbose = True
        shared_rng = self.ert.rng()
        ensemble_config = self.ert.ensembleConfig()

        return _lib.update.smoother_update(
            updatestep,
            total_ens_size,
            obs,
            shared_rng,
            analysis_config,
            ensemble_config,
            source_fs,
            target_fs,
            verbose,
        )
