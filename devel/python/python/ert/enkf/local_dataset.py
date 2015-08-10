from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.ecl import EclRegion

class LocalDataset(BaseCClass):

    def __init__(self, name):
        raise NotImplementedError("Class can not be instantiated directly!")
    
    def __contains__(self , key):
        return LocalDataset.cNamespace().has_key(self, key)
                 
    def getName(self):
        """ @rtype: str """
        return LocalDataset.cNamespace().name(self)
    
    def getActiveList(self, key):
        """ @rtype: ActiveList """
        if key in self:
            return LocalDataset.cNamespace().active_list(self , key)
        else:
            raise KeyError("Local key:%s not recognized" % key)

    def addNodeWithIndex(self, key, index):
        assert isinstance(key, str)
        assert isinstance(index, int)
        
        LocalDataset.cNamespace().add_node(self, key)              
        active_list = self.getActiveList(key)
        active_list.addActiveIndex(index)
                
    def addField(self, key, ecl_region):
        assert isinstance(key, str)
        assert isinstance(ecl_region, EclRegion)
        
        LocalDataset.cNamespace().add_node(self, key)
        active_list = self.getActiveList(key)
        
        active_region = ecl_region.getActiveList()
        
        for i in active_region:
            active_list.addActiveIndex(i)
                            
    def free(self):
        LocalDataset.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerObjectType("local_dataset", LocalDataset)

LocalDataset.cNamespace().alloc          = cwrapper.prototype("c_void_p local_dataset_alloc(char*)")
LocalDataset.cNamespace().has_key        = cwrapper.prototype("bool local_dataset_has_key(local_dataset, char*)")
LocalDataset.cNamespace().free           = cwrapper.prototype("void local_dataset_free(local_dataset)")
LocalDataset.cNamespace().name           = cwrapper.prototype("char* local_dataset_get_name(local_dataset)")
LocalDataset.cNamespace().active_list    = cwrapper.prototype("active_list_ref local_dataset_get_node_active_list(local_dataset, char*)")
LocalDataset.cNamespace().add_node       = cwrapper.prototype("void local_dataset_add_node(local_dataset, char*)")


                                                                                 
                                                                                 



