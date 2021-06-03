from cwrap import BaseCClass
from res import ResPrototype


class ESUpdate(BaseCClass):
    TYPE_NAME = "es_update"
    _smoother_update = ResPrototype(
        "bool enkf_main_smoother_update(es_update, enkf_fs, enkf_fs)"
    )

    def __init__(self, enkf_main):
        assert isinstance(enkf_main, BaseCClass)

        # enkf_main should be an EnKFMain, get the _RealEnKFMain object
        real_enkf_main = enkf_main.parent()

        super(ESUpdate, self).__init__(
            real_enkf_main.from_param(real_enkf_main).value,
            parent=real_enkf_main,
            is_reference=True,
        )

    def _analysis_config(self):
        return self.parent().analysisConfig()

    def hasModule(self, name):
        """
        Will check if we have analysis module @name.
        """
        return self._analysis_config().hasModule(name)

    def getModule(self, name):
        if self.hasModule(name):
            self._analysis_config().getModule(name)
        else:
            raise KeyError("No such module:%s " % name)

    def setGlobalStdScaling(self, weight):
        self._analysis_config().setGlobalStdScaling(weight)

    def smootherUpdate(self, run_context):
        data_fs = run_context.get_sim_fs()
        target_fs = run_context.get_target_fs()
        return self._smoother_update(data_fs, target_fs)
