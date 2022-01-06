
#include <ert/analysis/ies/ies_data.hpp>

void test_create() {
    auto *config = ies::config_alloc();

    ies::config_free(config);
}

int main(int argc, char **argv) { test_create(); }
