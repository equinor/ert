
#include <ert/analysis/ies/ies_enkf_data.hpp>

void test_create() {
    auto *config = ies::enkf_config_alloc();

    ies::enkf_config_free(config);
}

int main(int argc, char **argv) { test_create(); }
