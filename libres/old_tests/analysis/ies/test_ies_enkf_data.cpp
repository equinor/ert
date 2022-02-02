#include <ert/util/test_util.hpp>

#include <ert/util/rng.h>

#include <ert/analysis/ies/ies_data.hpp>

void test_create() {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    auto *data = static_cast<ies::data::data_type *>(ies::data::alloc(true));
    test_assert_not_NULL(data);
    ies::data::free(data);
    rng_free(rng);
}

int main(int argc, char **argv) { test_create(); }
