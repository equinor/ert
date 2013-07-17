/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_obs_vector_tests.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
 */

#include <ert/util/test_util.h>
#include <ert/enkf/obs_vector.h>
#include <ert/enkf/summary_obs.h>

bool alloc_strippedparameters_noerrors() {
  obs_vector_type * obs_vector = obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 0);
  obs_vector_free(obs_vector);
  return true;
}

bool scale_std_summary_nodata_no_errors() {
  obs_vector_type * obs_vector = obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 0);
  obs_vector_scale_std(obs_vector, 2.0);
  obs_vector_free(obs_vector);
  return true;
}

bool scale_std_summarysingleobservation_no_errors() {
  obs_vector_type * obs_vector = obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 1);
  
  summary_obs_type * summary_obs = summary_obs_alloc( "SummaryKey" , "ObservationKey" , 43.2, 2.0 , AUTO_CORRF_EXP, 42);
  obs_vector_install_node( obs_vector , 0 , summary_obs );

  test_assert_double_equal(2.0, summary_obs_get_std(summary_obs));
  obs_vector_scale_std(obs_vector, 2.0);
  test_assert_double_equal(4.0, summary_obs_get_std(summary_obs));

  obs_vector_free(obs_vector);
  return true;
}

bool scale_std_summarymanyobservations_no_errors() {
  double scaling_factor = 1.456;
  
  obs_vector_type * obs_vector = obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 100);
  
  test_assert_bool_equal(0, obs_vector_get_num_active(obs_vector));
  
  summary_obs_type* observations[100];
  for (int i=0; i<100; i++) {
    summary_obs_type * summary_obs = summary_obs_alloc( "SummaryKey" , "ObservationKey" , 43.2, i , AUTO_CORRF_EXP, 42);
    obs_vector_install_node( obs_vector , i , summary_obs );
    observations[i] = summary_obs;
  }
  
  for (int i=0; i<100; i++) {
    summary_obs_type * before_scale = observations[i];
    test_assert_double_equal(i, summary_obs_get_std(before_scale));
  }
  
  test_assert_bool_equal(100, obs_vector_get_num_active(obs_vector));

  obs_vector_scale_std(obs_vector, scaling_factor);
    
  for (int i=0; i<100; i++) {
    summary_obs_type * after_scale = observations[i];
    test_assert_double_equal(i * scaling_factor, summary_obs_get_std(after_scale));
  }
  
  obs_vector_free(obs_vector);
  return true;
}

int main(int argc, char ** argv) {
  test_assert_bool_equal(alloc_strippedparameters_noerrors(), true);
  test_assert_bool_equal(scale_std_summary_nodata_no_errors(), true);
  test_assert_bool_equal(scale_std_summarysingleobservation_no_errors(), true);
  test_assert_bool_equal(scale_std_summarymanyobservations_no_errors(), true);

  exit(0);
}

