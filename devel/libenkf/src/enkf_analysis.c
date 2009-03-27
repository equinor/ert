#include <stdlib.h>
#include <matrix.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <enkf_obs.h>
#include <enkf_state.h>
#include <enkf_node.h>
#include <thread_pool.h>



matrix_type * enkf_analysis_allocX() {
  matrix_type * X = matrix_alloc(10,10);
  return X;
}
