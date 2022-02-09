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

        time_map = source_fs.getTimeMap()

        last_step = time_map.getLastStep()
        if last_step < 0:
            last_step = self.ert.getModelConfig().get_last_history_restart()

        step_list = list(range(last_step + 1))

        local_config = self.ert.getLocalConfig()
        updatestep = local_config.getUpdatestep()

        analysis_config = self.ert.analysisConfig()

        total_ens_size = self.ert.getEnsembleSize()
        obs = self.ert.getObservations()
        verbose = True
        shared_rng = self.ert.rng()
        ensemble_config = self.ert.ensembleConfig()

        ok = _lib.update.smoother_update(
            step_list,
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

        return ok
