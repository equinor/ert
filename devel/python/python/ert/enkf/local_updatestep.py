from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB, LocalMinistep

class LocalUpdateStep(BaseCClass):

    def __init__(self, updatestep_key):
        raise NotImplementedError("Class can not be instantiated directly!")
          
    def attachMinistep(self, ministep):
        assert isinstance(ministep, LocalMinistep)
        LocalUpdateStep.cNamespace().attach_ministep(self,ministep)
                    
    def getName(self):
        """ @rtype: str """
        return LocalUpdateStep.cNamespace().name(self)
                       
    def free(self):
        LocalUpdateStep.cNamespace().free(self) 

cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("local_updatestep", LocalUpdateStep)
cwrapper.registerType("local_updatestep_obj", LocalUpdateStep.createPythonObject)
cwrapper.registerType("local_updatestep_ref", LocalUpdateStep.createCReference)

LocalUpdateStep.cNamespace().alloc               = cwrapper.prototype("c_void_p local_updatestep_alloc(char*)")
LocalUpdateStep.cNamespace().free                = cwrapper.prototype("void local_updatestep_free(local_updatestep)")
LocalUpdateStep.cNamespace().attach_ministep     = cwrapper.prototype("void local_updatestep_add_ministep(local_updatestep,local_ministep)")
LocalUpdateStep.cNamespace().name                = cwrapper.prototype("char* local_updatestep_get_name(local_updatestep)")



