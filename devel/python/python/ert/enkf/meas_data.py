from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.enkf.obs_data import ObsData
from ert.util import Matrix, IntVector


class MeasData(BaseCClass):

    def __init__(self, ens_mask):
        c_pointer = MeasData.cNamespace().alloc(ens_mask)
        super(MeasData, self).__init__(c_pointer)


    def createS(self):
        """ @rtype: Matrix """
        if self.activeObsSize() > 0:
            return MeasData.cNamespace().allocS(self)
        else:
            raise ValueError("No active observations - can not create S")

    def deactivateZeroStdSamples(self, obs_data):
        assert isinstance(obs_data, ObsData)
        self.cNamespace().deactivate_outliers(obs_data, self)


    def addBlock(self , obs_key , report_step , obs_size):
        return MeasData.cNamespace().add_block( self , obs_key , report_step , obs_size )

        
    def activeObsSize(self):
        return MeasData.cNamespace().get_active_obs_size( self )

    
    def getActiveEnsSize(self):
        return MeasData.cNamespace().get_active_ens_size(self)


    def getTotalEnsSize(self):
        return MeasData.cNamespace().get_total_ens_size(self)



    def free(self):
        MeasData.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerObjectType("meas_data", MeasData)

MeasData.cNamespace().alloc    = cwrapper.prototype("c_void_p meas_data_alloc(bool_vector)")
MeasData.cNamespace().free     = cwrapper.prototype("void meas_data_free(meas_data)")
MeasData.cNamespace().get_active_obs_size = cwrapper.prototype("int meas_data_get_active_obs_size(meas_data)")
MeasData.cNamespace().get_active_ens_size = cwrapper.prototype("int meas_data_get_active_ens_size( meas_data )")
MeasData.cNamespace().get_total_ens_size = cwrapper.prototype("int meas_data_get_total_ens_size( meas_data )")
MeasData.cNamespace().allocS    = cwrapper.prototype("matrix_obj meas_data_allocS(meas_data)")
MeasData.cNamespace().add_block = cwrapper.prototype("meas_block_ref meas_data_add_block(meas_data, char* , int , int)")

MeasData.cNamespace().deactivate_outliers  = cwrapper.prototype("void enkf_analysis_deactivate_std_zero(obs_data, meas_data)")



