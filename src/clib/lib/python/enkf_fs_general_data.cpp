#include <ert/enkf/enkf_plot_gendata.hpp>
#include <ert/python.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

ERT_CLIB_SUBMODULE("enkf_fs_general_data", m) {
    m.def(
        "gendata_get_realizations",
        [](py::object self, const std::vector<int> &realizations) {
            auto enkf_plot_gendata =
                ert::from_cwrap<enkf_plot_gendata_type>(self);
            const int data_size =
                enkf_plot_gendata_get_data_size(enkf_plot_gendata);
            if (data_size < 0) {
                throw pybind11::value_error(
                    "No data has been loaded for report step");
            }
            auto active_mask = enkf_plot_gendata_active_mask(enkf_plot_gendata);
            const int realization_size = std::size(realizations);
            const size_t size = data_size * realization_size;

            double *data = new double[size];
            std::fill_n(data, size, NAN);

            for (int realization_index = 0;
                 realization_index < realization_size; realization_index++) {
                int realization = realizations.at(realization_index);
                enkf_plot_genvector_type *vector =
                    enkf_plot_gendata_iget(enkf_plot_gendata, realization);
                int current_data_size = enkf_plot_genvector_get_size(vector);
                // Following comment is from orginal code
                // see: https://github.com/equinor/ert/blob/3f84b979b985376aade9203a6b977b9b541d8c14/res/enkf/export/gen_data_collector.py#L46
                // Must check because of a bug changing between different case with different states
                if (current_data_size > 0) {
                    int data_index;
                    for (data_index = 0; data_index < current_data_size;
                         data_index++) {
                        if (active_mask.at(data_index)) {
                            data[realization_index +
                                 realization_size * data_index] =
                                enkf_plot_genvector_iget(vector, data_index);
                        }
                    }
                }
            }

            py::capsule free_when_done(data, [](void *f) {
                double *data = reinterpret_cast<double *>(f);
                delete[] data;
            });

            return py::array_t<double>(
                {data_size, realization_size}, // shape
                {realization_size * sizeof(double),
                 sizeof(double)}, // C-style contiguous strides for double
                data,             // the data pointer
                free_when_done);  // numpy array references this parent
        },
        py::arg("self"), py::arg("realizations"));
}
