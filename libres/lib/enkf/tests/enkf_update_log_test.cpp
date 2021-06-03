#include <stdio.h>
#include <string.h>

#include <ert/util/test_util.h>
#include <ert/util/int_vector.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/enkf_analysis.hpp>


void test_obs_in_log() {
  int_vector_type * ens_active_list = int_vector_alloc(0 , false);
  bool_vector_type * ens_mask;
  int_vector_append( ens_active_list , 10 );

  ens_mask = int_vector_alloc_mask(ens_active_list);
  meas_data_type * meas_data = meas_data_alloc( ens_mask );
  meas_block_type * block = meas_data_add_block(meas_data , "OBS" , 10 , 10);

  obs_data_type * obs_data = obs_data_alloc(1.0);
  obs_block_type * obs_block = obs_data_add_block( obs_data , "OBS", 10 , NULL , false );
  for (int iobs = 0; iobs < 10; iobs++)
    obs_block_iset( obs_block , iobs , iobs , 0.1);
  /* Set one obs block as missing*/
  obs_block_iset_missing(obs_block, 0);

  const char * ministep_name;
  ministep_name = "A random name";

  /* Check that the update_log contains the first observation name */
  char *str_stream;
  size_t size;
  FILE *stream;
  stream = open_memstream (&str_stream, &size);
  enkf_analysis_fprintf_obs_summary(obs_data, meas_data, ens_active_list, ministep_name, stream);
  fclose(stream);
  test_assert_true((strstr(str_stream, "OBS") != NULL));
  test_assert_false((strstr(str_stream, "* Local inactive") != NULL));

  free(str_stream);
  obs_data_free( obs_data );
  meas_data_free( meas_data );
  bool_vector_free( ens_mask );
  int_vector_free( ens_active_list );
}


void test_local_inactive() {
  int_vector_type * ens_active_list = int_vector_alloc(0 , false);
  bool_vector_type * ens_mask;
  int_vector_append( ens_active_list , 10 );

  ens_mask = int_vector_alloc_mask(ens_active_list);
  meas_data_type * meas_data = meas_data_alloc( ens_mask );
  meas_block_type * block = meas_data_add_block(meas_data , "OBS" , 10 , 10);

  obs_data_type * obs_data = obs_data_alloc(1.0);
  obs_block_type * obs_block = obs_data_add_block( obs_data , "OBS", 10 , NULL , false );

  /* By not setting all obs we set one obs block as local inactive*/
  for (int iobs = 1; iobs < 10; iobs++)
    obs_block_iset( obs_block , iobs , iobs , 0.1);


  const char * ministep_name;
  ministep_name = "A random name";

  /* Check that the update_log contains the first observation name */
  char *str_stream;
  size_t size;
  FILE *stream;
  stream = open_memstream (&str_stream, &size);
  enkf_analysis_fprintf_obs_summary(obs_data, meas_data, ens_active_list, ministep_name, stream);
  fclose(stream);

  test_assert_true((strstr(str_stream, "OBS") != NULL));
  test_assert_true((strstr(str_stream, "* Local inactive") != NULL));


  free(str_stream);
  obs_data_free( obs_data );
  meas_data_free( meas_data );
  bool_vector_free( ens_mask );
  int_vector_free( ens_active_list );
}


int main(int argc , char ** argv) {
  test_obs_in_log();
  test_local_inactive();
  exit(0);
}

