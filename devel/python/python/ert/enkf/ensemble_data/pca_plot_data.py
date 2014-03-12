from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.util import Matrix
from ert.enkf.ensemble_data import PcaPlotVector


class PcaPlotData(BaseCClass):

    def __init__(self, name, principal_component_matrix, observation_principal_component_matrix):
        assert isinstance(name, str)
        assert isinstance(principal_component_matrix, Matrix)
        assert isinstance(observation_principal_component_matrix, Matrix)

        c_pointer = PcaPlotData.cNamespace().alloc(name, principal_component_matrix, observation_principal_component_matrix)
        super(PcaPlotData, self).__init__(c_pointer)


    def __len__(self):
        """ @rtype: int """
        return PcaPlotData.cNamespace().component_count(self)


    def __getitem__(self, index):
        """ @rtype: PcaPlotVector """
        assert isinstance(index, int)
        return PcaPlotData.cNamespace().get(self, index).setParent(self)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def free(self):
        PcaPlotData.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("pca_plot_data", PcaPlotData)
cwrapper.registerType("pca_plot_data_obj", PcaPlotData.createPythonObject)
cwrapper.registerType("pca_plot_data_ref", PcaPlotData.createCReference)

PcaPlotData.cNamespace().alloc             = cwrapper.prototype("c_void_p pca_plot_data_alloc(char*, matrix, matrix)")
PcaPlotData.cNamespace().free              = cwrapper.prototype("void pca_plot_data_free(pca_plot_data)")

PcaPlotData.cNamespace().component_count   = cwrapper.prototype("int pca_plot_data_get_size(pca_plot_data)")
PcaPlotData.cNamespace().realization_count = cwrapper.prototype("int pca_plot_data_get_ens_size(pca_plot_data)")
PcaPlotData.cNamespace().get               = cwrapper.prototype("pca_plot_vector_ref pca_plot_data_iget_vector(pca_plot_data, int)")
PcaPlotData.cNamespace().get_name          = cwrapper.prototype("char* pca_plot_data_get_name(pca_plot_data)")
