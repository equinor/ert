
#include <ert/analysis/ies/ies_data.hpp>

void test_create() {
    auto *config = ies::config::alloc(true);

    ies::config::free(config);
}

int main(int argc, char **argv) { test_create(); }
