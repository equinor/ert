#include <ert/util/test_util.hpp>

#include <ert/util/rng.h>

#include "ies_enkf_data.h"
#include "ies_enkf_config.h"


void test_create() {
  rng_type * rng = rng_alloc( MZRAN, INIT_DEFAULT );
  ies_enkf_data_type * data = (ies_enkf_data_type *) ies_enkf_data_alloc(rng);
  test_assert_not_NULL(data);
  ies_enkf_data_free( data );
  rng_free( rng );
}


int main(int argc, char ** argv) {
  test_create();
}
