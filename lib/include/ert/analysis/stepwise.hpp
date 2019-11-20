#ifndef ERT_STEPWISE_H
#define ERT_STEPWISE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>

  typedef struct stepwise_struct stepwise_type;


  stepwise_type * stepwise_alloc1(int nsample, int nvar, rng_type * rng, const matrix_type* St, const matrix_type* Et);
  void            stepwise_free( stepwise_type * stepwise);

  void            stepwise_set_Y0( stepwise_type * stepwise ,  matrix_type * Y);
  void            stepwise_set_R2( stepwise_type * stepwise ,  const double R2);
  int             stepwise_get_n_active( stepwise_type * stepwise );
  bool_vector_type * stepwise_get_active_set( stepwise_type * stepwise );
  double          stepwise_iget_beta(const stepwise_type * stepwise, const int index );
  double          stepwise_get_sum_beta(const stepwise_type * stepwise );

  void            stepwise_estimate( stepwise_type * stepwise , double deltaR2_limit , int CV_blocks);
  double          stepwise_eval( const stepwise_type * stepwise , const matrix_type * x );



#ifdef __cplusplus
}
#endif

#endif
