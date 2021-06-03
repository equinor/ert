

#include "ies_enkf_data.h"


void test_create() {
  ies_enkf_config_type * config = ies_enkf_config_alloc();

  ies_enkf_config_free(config);
}

int main(int argc, char ** argv) {
  test_create();
}
