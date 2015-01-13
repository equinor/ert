from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.enkf.obs_data import ObsData
from ert.util import Matrix, IntVector


class MeasBlock(BaseCClass):

    def __init__(self , obs_key , obs_size , ens_size ):
        c_pointer = MeasBlock.cNamespace().alloc( obs_key , ens_size , obs_size )
        super(MeasBlock , self).__init__(c_pointer)

    def getObsSize(self):
        return MeasBlock.cNamespace().get_obs_size(self)


    def getEnsSize(self):
        return MeasBlock.cNamespace().get_ens_size(self)

        
    def __assert_index__(self , index):
        if isinstance(index , tuple):
            iobs,iens = index
            if not 0 <= iobs < self.getObsSize():
                raise IndexError("Invalid iobs value:%d  Valid range: [0,%d)" % (iobs , self.getObsSize()))
                
            if not 0 <= iens < self.getEnsSize():
                raise IndexError("Invalid iens value:%d  Valid range: [0,%d)" % (iobs , self.getEnsSize()))

            return iobs,iens
        else:
            raise TypeError("The index argument must be 2-tuple")


    def __setitem__(self, index, value):
        iobs , iens = self.__assert_index__(index)
        MeasBlock.cNamespace().iset_value( self , iens , iobs , value )


    def __getitem__(self, index):
        iobs,iens = self.__assert_index__(index)
        return MeasBlock.cNamespace().iget_value( self , iens , iobs )
        

    def free(self):
        MeasBlock.cNamespace().free(self)


    def igetMean(self , iobs):
        if 0 <= iobs < self.getObsSize():
            return MeasBlock.cNamespace().iget_mean(self , iobs)
        else:
            raise IndexError("Invalid observation index:%d  valid range: [0,%d)" % (iobs , self.getObsSize()))

    def igetStd(self , iobs):
        if 0 <= iobs < self.getObsSize():
            return MeasBlock.cNamespace().iget_std(self , iobs)
        else:
            raise IndexError("Invalid observation index:%d  valid range: [0,%d)" % (iobs , self.getObsSize()))



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerObjectType("meas_block", MeasBlock)

MeasBlock.cNamespace().alloc = cwrapper.prototype("c_void_p meas_block_alloc( char* , int , int)")
MeasBlock.cNamespace().free = cwrapper.prototype("void meas_block_free( meas_block )")
MeasBlock.cNamespace().get_ens_size = cwrapper.prototype("int meas_block_get_ens_size( meas_block )")
MeasBlock.cNamespace().get_obs_size = cwrapper.prototype("int meas_block_get_total_obs_size( meas_block )")
MeasBlock.cNamespace().iget_value = cwrapper.prototype("double meas_block_iget( meas_block , int , int)")
MeasBlock.cNamespace().iset_value = cwrapper.prototype("void meas_block_iset( meas_block , int , int , double)")
MeasBlock.cNamespace().iget_mean = cwrapper.prototype("double meas_block_iget_ens_mean( meas_block , int )")
MeasBlock.cNamespace().iget_std = cwrapper.prototype("double meas_block_iget_ens_std( meas_block , int )")


    


