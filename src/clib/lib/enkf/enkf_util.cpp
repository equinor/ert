#include <cmath>
#include <ert/ecl/ecl_util.h>
#include <ert/util/rng.h>
#include <ert/util/util.h>
#include <random>
#include <stdlib.h>

#include <ert/enkf/enkf_util.hpp>
#include <ert/python.hpp>

class generator {
    rng_type *rng;

public:
    generator(rng_type *rng) : rng(rng) {}

    using value_type = unsigned int;
    static constexpr value_type min() { return 0; }
    static constexpr value_type max() { return UINT32_MAX; }

    value_type operator()() { return rng_forward(rng); }
};

double enkf_util_rand_normal(double mean, double std, rng_type *rng) {
    generator gen(rng);
    std::normal_distribution<double> normdist{mean, std};
    return normdist(gen);
}

void enkf_util_assert_buffer_type(buffer_type *buffer,
                                  ert_impl_type target_type) {
    ert_impl_type file_type = INVALID;
    file_type = (ert_impl_type)buffer_fread_int(buffer);
    if (file_type != target_type)
        util_abort(
            "%s: wrong target type in file (expected:%d  got:%d) - aborting \n",
            __func__, target_type, file_type);
}

ERT_CLIB_SUBMODULE("enkf_util", m) {
    m.def("standard_normal", [](py::handle rng) {
        return enkf_util_rand_normal(0, 1, ert::from_cwrap<rng_type>(rng));
    });
}
