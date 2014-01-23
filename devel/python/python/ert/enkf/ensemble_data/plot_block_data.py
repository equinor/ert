from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB, EnkfStateType, EnkfFs
from ert.enkf.observations import ObsVector
from ert.enkf.ensemble_data import PlotBlockVector
from ert.util import DoubleVector, BoolVector


class PlotBlockData(BaseCClass):

    def __init__(self, obs_vector, file_system=None, report_step=None, state=EnkfStateType.FORECAST, input_mask=None):
        assert isinstance(obs_vector, ObsVector)

        c_pointer = PlotBlockData.cNamespace().alloc(obs_vector)
        super(PlotBlockData, self).__init__(c_pointer)

        if not file_system is None:
            self.load(file_system, report_step, state, input_mask)



    def load(self, file_system, report_step, state=EnkfStateType.FORECAST, input_mask=None):
        assert isinstance(file_system, EnkfFs)
        assert isinstance(report_step, int)
        assert isinstance(state, EnkfStateType)
        if not input_mask is None:
            assert isinstance(input_mask, BoolVector)

        PlotBlockData.cNamespace().load(self, file_system, report_step, state, input_mask)


    def __len__(self):
        """ @rtype: int """
        return PlotBlockData.cNamespace().size(self)


    def __getitem__(self, index):
        """ @rtype: PlotBlockVector """
        assert isinstance(index, int)
        return PlotBlockData.cNamespace().get(self, index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def getDepth(self):
        """ @rtype: DoubleVector """
        return PlotBlockData.cNamespace().get_depth(self)


    def free(self):
        PlotBlockData.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("plot_block_data", PlotBlockData)
cwrapper.registerType("plot_block_data_obj", PlotBlockData.createPythonObject)
cwrapper.registerType("plot_block_data_ref", PlotBlockData.createCReference)

PlotBlockData.cNamespace().free = cwrapper.prototype("void enkf_plot_blockdata_free(plot_block_data)")
PlotBlockData.cNamespace().alloc = cwrapper.prototype("c_void_p enkf_plot_blockdata_alloc(obs_vector)")
PlotBlockData.cNamespace().size = cwrapper.prototype("int enkf_plot_blockdata_get_size(obs_vector)")
PlotBlockData.cNamespace().get = cwrapper.prototype("plot_block_vector_ref enkf_plot_blockdata_iget(plot_block_data, int)")
PlotBlockData.cNamespace().get_depth = cwrapper.prototype("double_vector_ref enkf_plot_blockdata_get_depth(plot_block_data)")
PlotBlockData.cNamespace().load = cwrapper.prototype("void enkf_plot_blockdata_load(plot_block_data, enkf_fs, int, enkf_state_type_enum, bool_vector)")


