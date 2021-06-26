#include <ert/util/test_util.hpp>

#include "ies_enkf_data.h"


void test_create() {
  ies_enkf_data_type * data = (ies_enkf_data_type *) ies_enkf_data_alloc();
  test_assert_not_NULL(data);
  ies_enkf_data_free( data );
}


int main(int argc, char ** argv) {
  test_create();
}
