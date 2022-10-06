from cwrap import BaseCEnum


class EnkfObservationImplementationType(BaseCEnum):
    TYPE_NAME = "enkf_obs_impl_type"
    GEN_OBS = None
    SUMMARY_OBS = None


EnkfObservationImplementationType.addEnum("GEN_OBS", 1)
EnkfObservationImplementationType.addEnum("SUMMARY_OBS", 2)
