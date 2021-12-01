#include <ert/util/test_util.hpp>

#include <ert/util/rng.h>

#include <ert/analysis/ies/ies_enkf_state.hpp>

void test_create() {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    ies_enkf_state_type *data = (ies_enkf_state_type *)ies_enkf_state_alloc();
    test_assert_not_NULL(data);
    ies_enkf_state_free(data);
    rng_free(rng);
}

int main(int argc, char **argv) { test_create(); }
