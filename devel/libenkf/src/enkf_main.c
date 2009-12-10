#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <enkf_config_node.h>
#include <ecl_util.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <thread_pool.h>
#include <obs_data.h>
#include <history.h>
#include <meas_matrix.h>
#include <enkf_state.h>  
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <arg_pack.h>
#include <history.h>
#include <node_ctype.h>
#include <pthread.h>
#include <job_queue.h>
#include <msg.h>
#include <stringlist.h>
#include <enkf_main.h> 
#include <enkf_serialize.h>
#include <config.h>  
#include <local_driver.h>
#include <rsh_driver.h>
#include <lsf_driver.h>
#include <history.h>
#include <enkf_sched.h>
#include <set.h>
#include <ecl_io_config.h>
#include <ecl_config.h>
#include <ensemble_config.h>
#include <model_config.h>
#include <site_config.h>
#include <active_config.h>
#include <forward_model.h>
#include <enkf_analysis.h>
#include <local_ministep.h>
#include <local_updatestep.h>
#include <local_config.h>
#include <misfit_table.h>
#include <log.h>
#include <plot_config.h>
#include <ert_template.h>
#include <dirent.h>
#include <pwd.h>
#include <unistd.h>
#include <sys/types.h>
#include <job_queue.h>
#include <basic_queue_driver.h>
#include <subst_func.h>
#include "enkf_defaults.h"

/**
   This object should contain **everything** needed to run a enkf
   simulation. A way to wrap up all available information/state and
   pass it around. An attempt has been made to collect various pieces
   of related information together in a couple of objects
   (model_config, ecl_config, site_config and ensemble_config). When
   it comes to these holding objects the following should be observed:

    1. It not always obvious where a piece of information should be
       stored, i.e. the grid is a property of the model, however it is
       an eclipse grid, and hence also belongs to eclipse
       configuration?? [In this case ecl_config wins out.]

    2. The information stored in these objects is typically passed on
       to the enkf_state object, where it is used. 

    3. At enkf_state level it is not really consequent - in some cases
       the enkf_state object takes a scalar copy (i.e. keep_runpath),
       and in other cases only a pointer down to the underlying
       enkf_main object is taken. In the former case it is no way to
       change global behaviour by modifying the enkf_main objects.
       
       In the enkf_state object the fields of the member_config,
       ecl_config, site_config and ensemble_config objects are mixed
       and matched into other small holding objects defined in
       enkf_state.c.

*/

#define ENKF_MAIN_ID              8301
#define ENKF_ENSEMBLE_TYPE_ID   776098

//struct enkf_ensemble_struct {
//  
//};



struct enkf_main_struct {
  UTIL_TYPE_ID_DECLARATION;
  ensemble_config_type * ensemble_config;  /* The config objects for the various enkf nodes.*/ 
  enkf_fs_type         * dbase;            /* The internalized information. */
  model_config_type    * model_config;
  ecl_config_type      * ecl_config;
  site_config_type     * site_config;
  analysis_config_type * analysis_config;
  local_config_type    * local_config;     /* Holding all the information about local analysis. */
  log_type             * logh;             /* Handle to an open log file. */
  plot_config_type     * plot_config;      /* Information about plotting. */
  ert_templates_type   * templates;       
  subst_func_pool_type * subst_func_pool;
  subst_list_type      * subst_list;       /* A parent subst_list instance - common to all ensemble members. */
  /*-------------------------*/

  keep_runpath_type    * keep_runpath;     /* HACK: This is only used in the initialization period - afterwards the data is held by the enkf_state object. */
  
  
  enkf_obs_type        * obs;               
  misfit_table_type    * misfit_table;     /* An internalization of misfit results - used for ranking according to various criteria. */
  enkf_state_type     ** ensemble;         /* The ensemble ... */
}; 




/*****************************************************************/

void enkf_main_init_internalization( enkf_main_type *  , run_mode_type  );

/*****************************************************************/

SAFE_CAST(enkf_main , ENKF_MAIN_ID)


ensemble_config_type * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main) {
  return enkf_main->ensemble_config;
}

model_config_type * enkf_main_get_model_config( const enkf_main_type * enkf_main ) {
  return enkf_main->model_config;
}

plot_config_type * enkf_main_get_plot_config( const enkf_main_type * enkf_main ) {
  return enkf_main->plot_config;
}

ecl_config_type *enkf_main_get_ecl_config(const enkf_main_type * enkf_main) {
	return enkf_main->ecl_config;
}

int enkf_main_get_history_length( const enkf_main_type * enkf_main) {
  return model_config_get_last_history_restart( enkf_main->model_config);
}

bool enkf_main_has_prediction( const enkf_main_type * enkf_main ) {
  return model_config_has_prediction( enkf_main->model_config );
}

enkf_fs_type * enkf_main_get_fs(const enkf_main_type * enkf_main) {
  return enkf_main->dbase;
}


enkf_obs_type * enkf_main_get_obs(const enkf_main_type * enkf_main) {
  return enkf_main->obs;
}


misfit_table_type * enkf_main_get_misfit(const enkf_main_type * enkf_main) {
  return enkf_main->misfit_table;
}



void enkf_main_free(enkf_main_type * enkf_main) {  
  enkf_obs_free(enkf_main->obs);
  {
    const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
    int i;
    for (i=0; i < ens_size; i++)
      enkf_state_free(enkf_main->ensemble[i]);
    free(enkf_main->ensemble);
  }
  if (enkf_main->dbase != NULL) enkf_fs_free( enkf_main->dbase );
  
  log_add_message( enkf_main->logh , false , NULL , "Exiting ert application normally - all is fine(?)" , false);
  log_close( enkf_main->logh );
  analysis_config_free(enkf_main->analysis_config);
  ecl_config_free(enkf_main->ecl_config);
  model_config_free( enkf_main->model_config);
  site_config_free( enkf_main->site_config);
  ensemble_config_free( enkf_main->ensemble_config );
  local_config_free( enkf_main->local_config );
  if (enkf_main->misfit_table != NULL)
    misfit_table_free( enkf_main->misfit_table );
  util_safe_free( enkf_main->keep_runpath );
  plot_config_free( enkf_main->plot_config );
  ert_templates_free( enkf_main->templates );

  subst_func_pool_free( enkf_main->subst_func_pool );
  subst_list_free( enkf_main->subst_list );

  free(enkf_main);
}



/*****************************************************************/



static void enkf_main_load_sub_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state, int iens1 , int iens2) {
  int iens;
  for (iens = iens1; iens < iens2; iens++)
    enkf_state_fread(enkf_main->ensemble[iens] , mask , report_step , state );
}


static void * enkf_main_load_sub_ensemble__(void * __arg) {
  arg_pack_type * arg_pack   = arg_pack_safe_cast(__arg);
  enkf_main_type * enkf_main = arg_pack_iget_ptr(arg_pack , 0);
  int mask                   = arg_pack_iget_int(arg_pack , 1);
  int report_step            = arg_pack_iget_int(arg_pack , 2);
  state_enum state           = arg_pack_iget_int(arg_pack , 3);
  int iens1                  = arg_pack_iget_int(arg_pack , 4);
  int iens2                  = arg_pack_iget_int(arg_pack , 5);

  enkf_main_load_sub_ensemble(enkf_main , mask , report_step , state , iens1 , iens2);
  return NULL;
}



void enkf_main_load_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  const   int cpu_threads = 4;
  int     sub_ens_size    = ensemble_config_get_size(enkf_main->ensemble_config) / cpu_threads;
  int     icpu;
  thread_pool_type * tp          = thread_pool_alloc( cpu_threads );
  arg_pack_type ** arg_pack_list = util_malloc( cpu_threads * sizeof * arg_pack_list , __func__);
  
  for (icpu = 0; icpu < cpu_threads; icpu++) {
    arg_pack_type * arg = arg_pack_alloc();
    arg_pack_append_ptr(arg , enkf_main);
    arg_pack_append_int(arg , mask);
    arg_pack_append_int(arg , report_step);
    arg_pack_append_int(arg , state);
    
    {
      int iens1 =  icpu * sub_ens_size;
      int iens2 = iens1 + sub_ens_size;
      
      if (icpu == (cpu_threads - 1))
	iens2 = ensemble_config_get_size(enkf_main->ensemble_config);

      arg_pack_append_int(arg ,  iens1);
      arg_pack_append_int(arg ,  iens2);
    }
    arg_pack_list[icpu] = arg;
    arg_pack_lock( arg );
    thread_pool_add_job( tp , enkf_main_load_sub_ensemble__ , arg);
  }
  thread_pool_join( tp );
  thread_pool_free( tp );

  for (icpu = 0; icpu < cpu_threads; icpu++) 
    arg_pack_free( arg_pack_list[icpu] );
  free(arg_pack_list);
}





static void enkf_main_fwrite_sub_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state, int iens1 , int iens2) {
  int iens;
  for (iens = iens1; iens < iens2; iens++)
    enkf_state_fwrite(enkf_main->ensemble[iens] , mask , report_step , state);
}


static void * enkf_main_fwrite_sub_ensemble__(void *__arg) {
  arg_pack_type * arg_pack   = arg_pack_safe_cast(__arg);
  enkf_main_type * enkf_main = arg_pack_iget_ptr(arg_pack , 0);
  int mask                   = arg_pack_iget_int(arg_pack , 1);
  int report_step            = arg_pack_iget_int(arg_pack , 2);
  state_enum state           = arg_pack_iget_int(arg_pack , 3);
  int iens1                  = arg_pack_iget_int(arg_pack , 4);
  int iens2                  = arg_pack_iget_int(arg_pack , 5);

  enkf_main_fwrite_sub_ensemble(enkf_main , mask , report_step , state , iens1 , iens2);
  return NULL;
}


void enkf_main_fwrite_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  const   int cpu_threads = 4;
  int     sub_ens_size    = ensemble_config_get_size(enkf_main->ensemble_config) / cpu_threads;
  int     icpu;
  thread_pool_type * tp = thread_pool_alloc( cpu_threads );
  arg_pack_type ** arg_pack_list = util_malloc( cpu_threads * sizeof * arg_pack_list , __func__);
  
  for (icpu = 0; icpu < cpu_threads; icpu++) {
    arg_pack_type * arg = arg_pack_alloc();
    arg_pack_append_ptr(arg , enkf_main);
    arg_pack_append_int(arg , mask);
    arg_pack_append_int(arg , report_step);
    arg_pack_append_int(arg , state);
    
    {
      int iens1 =  icpu * sub_ens_size;
      int iens2 = iens1 + sub_ens_size;
      
      if (icpu == (cpu_threads - 1))
	iens2 = ensemble_config_get_size(enkf_main->ensemble_config);

      arg_pack_append_int(arg , iens1);
      arg_pack_append_int(arg , iens2);
    }
    arg_pack_list[icpu] = arg;
    arg_pack_lock( arg );
    thread_pool_add_job( tp , enkf_main_fwrite_sub_ensemble__ , arg);
  }
  thread_pool_join( tp );
  thread_pool_free( tp );

  for (icpu = 0; icpu < cpu_threads; icpu++) 
    arg_pack_free( arg_pack_list[icpu]);
  free(arg_pack_list);
}




/**
   This function returns a (enkf_node_type ** ) pointer, which points
   to all the instances with the same keyword, i.e.

   enkf_main_get_node_ensemble(enkf_main , "PRESSURE");
  
   Will return an ensemble of pressure nodes. Observe that apart from
   the list of pointers, *now new storage* is allocated, all the
   pointers point in to the underlying enkf_node instances under the
   enkf_main / enkf_state objects. Consequently there is no designated
   free() function to match this, just free() the result.

   Example:
   
   enkf_node_type ** pressure_nodes = enkf_main_get_node_ensemble(enkf_main , "PRESSURE");
 
   Do something with the pressure nodes ... 

   free(pressure_nodes);

*/

static enkf_node_type ** enkf_main_get_node_ensemble(const enkf_main_type * enkf_main , const char * key , int report_step , state_enum load_state) {
  enkf_fs_type * fs               = enkf_main_get_fs( enkf_main );
  const int ens_size              = ensemble_config_get_size(enkf_main->ensemble_config);
  enkf_node_type ** node_ensemble = util_malloc(ens_size * sizeof * node_ensemble , __func__ );
  int iens;
  
  for (iens = 0; iens < ens_size; iens++) {
    node_ensemble[iens] = enkf_state_get_node(enkf_main->ensemble[iens] , key);
    enkf_fs_fread_node( fs , node_ensemble[iens] , report_step , iens ,load_state);
  }
  return node_ensemble;
}

/*****************************************************************/




enkf_state_type * enkf_main_iget_state(const enkf_main_type * enkf_main , int iens) {
  return enkf_main->ensemble[iens];
}


member_config_type * enkf_main_iget_member_config(const enkf_main_type * enkf_main , int iens) {
  return enkf_state_get_member_config( enkf_main->ensemble[iens] );
}



void enkf_main_node_mean( const enkf_node_type ** ensemble , int ens_size , enkf_node_type * mean ) {
  int iens;
  enkf_node_clear( mean );
  for (iens = 0; iens < ens_size; iens++) 
    enkf_node_iadd( mean , ensemble[iens] );
    
  enkf_node_scale( mean , 1.0 / ens_size );
}


/**
   This function calculates the node standard deviation from the
   ensemble. The mean can be NULL, in which case it is assumed that
   the mean has already been shifted away from the ensemble. 
*/


void enkf_main_node_std( const enkf_node_type ** ensemble , int ens_size , const enkf_node_type * mean , enkf_node_type * std) {
  int iens;
  enkf_node_clear( std );
  for (iens = 0; iens < ens_size; iens++) 
    enkf_node_iaddsqr( std , ensemble[iens] );
  enkf_node_scale(std , 1.0 / ens_size );

  if (mean != NULL) {
    enkf_node_scale( std , -1 );
    enkf_node_iaddsqr( std , mean );
    enkf_node_scale( std , -1 );
  }

  enkf_node_sqrt( std );
}


void enkf_main_inflate_node(enkf_main_type * enkf_main , int report_step , const char * key , const enkf_node_type * min_std) {
  int ens_size                              = ensemble_config_get_size(enkf_main->ensemble_config);  
  enkf_node_type ** ensemble                = enkf_main_get_node_ensemble( enkf_main , key , report_step , ANALYZED );
  enkf_node_type * mean                     = enkf_node_copyc( ensemble[0] );
  enkf_node_type * std                      = enkf_node_copyc( ensemble[0] );
  int iens;
  
  enkf_main_node_mean( (const enkf_node_type **) ensemble , ens_size , mean );
  /* Shifting away the mean */
  enkf_node_scale( mean , -1 );
  for (iens = 0; iens < ens_size; iens++) 
    enkf_node_iadd( ensemble[iens] , mean );
  enkf_node_scale( mean , -1 );
  
  
  enkf_main_node_std( (const enkf_node_type **) ensemble , ens_size , NULL , std );
  /*****************************************************************/
  /*
    Now we have the ensemble represented as a mean and an ensemble of
    deviations from the mean. This is the form suitable for actually
    doing the inflation.
  */
  {
    enkf_node_type * inflation = enkf_node_copyc( ensemble[0] );
    enkf_node_set_inflation( inflation , std , min_std  );
    
    for (iens = 0; iens < ens_size; iens++) 
      enkf_node_imul( ensemble[iens] , inflation );
    
    enkf_node_free( inflation );
  }
  
  
  /* Add the mean back in - and store the updated node to disk.*/
  for (iens = 0; iens < ens_size; iens++) {
    enkf_node_iadd( ensemble[iens] , mean );
    enkf_fs_fwrite_node( enkf_main_get_fs( enkf_main ) , ensemble[iens] , report_step , iens , ANALYZED );
  }
  
  enkf_node_free( mean );
  enkf_node_free( std );
  free( ensemble );
}
  



void enkf_main_inflate(enkf_main_type * enkf_main , int report_step , hash_type * use_count) {
  stringlist_type * keys = ensemble_config_alloc_keylist_from_var_type( enkf_main->ensemble_config , PARAMETER + DYNAMIC_STATE);
  msg_type * msg = msg_alloc("Inflating:");

  msg_show( msg );
  for (int ikey = 0; ikey < stringlist_get_size( keys ); ikey++) {
    const char * key = stringlist_iget( keys  , ikey );
    if (hash_get_counter(use_count , key) > 0) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( enkf_main->ensemble_config , key );
      const enkf_node_type * min_std            = enkf_config_node_get_min_std( config_node );
      
      if (min_std != NULL) {
        msg_update( msg , key );
        enkf_main_inflate_node(enkf_main , report_step , key , min_std );
      }
    }
  }
  stringlist_free( keys );
  msg_free( msg , true );
}




static int __get_active_size(const enkf_config_node_type * config_node , int report_step , const active_list_type * active_list) {
  active_mode_type active_mode = active_list_get_mode( active_list );
  int active_size;
  if (active_mode == INACTIVE)
    active_size = 0;
  else if (active_mode == ALL_ACTIVE)
    active_size = enkf_config_node_get_data_size( config_node , report_step );
  else if (active_mode == PARTLY_ACTIVE)
    active_size = active_list_get_active_size( active_list );
  else {
    util_abort("%s: internal error .. \n",__func__); 
    active_size = -1; /* Compiler shut up */
  }
  return active_size;
}



void enkf_main_update_mulX(enkf_main_type * enkf_main , const matrix_type * X5 , const local_ministep_type * ministep, int report_step , hash_type * use_count) {

  int       matrix_size              = 1000;  /* Starting with this */
  const int ens_size                 = ensemble_config_get_size(enkf_main->ensemble_config);
  enkf_fs_type * fs                  = enkf_main_get_fs( enkf_main ); 
  matrix_type * A = matrix_alloc(matrix_size , ens_size);
  msg_type  * msg = msg_alloc("Updating: ");
  stringlist_type * update_keys = local_ministep_alloc_node_keys( ministep );
  const int num_kw  = stringlist_get_size( update_keys );
  int * active_size = util_malloc( num_kw * sizeof * active_size , __func__);
  int * row_offset  = util_malloc( num_kw * sizeof * row_offset  , __func__);
  int ikw           = 0;
  bool complete     = false;

  msg_show( msg );
  do {
    bool first_kw    	     = true;
    bool add_more_kw 	     = true;
    int ikw1 	     	     = ikw;
    int ikw2 	     	     = ikw;  
    int current_row_offset   = 0;
    matrix_resize( A , matrix_size , ens_size , false);  /* Recover full matrix size - after matrix_shrink_header() has been called. */
    do {
      const char             * key              = stringlist_iget(update_keys , ikw);
      const active_list_type * active_list      = local_ministep_get_node_active_list( ministep , key );
      const enkf_config_node_type * config_node = ensemble_config_get_node( enkf_main->ensemble_config , key );


      /** 
	  This is very awkward; the problem is that for the GEN_DATA
	  type the config object does not really own the size. Instead
	  the size is pushed (on load time) from gen_data instances to
	  the gen_data_config instance. Therefor we have to assert
	  that at least one gen_data instance has been loaded (and
	  consequently updated the gen_data_config instance) before we
	  query for the size.
      */
      {
	if (enkf_config_node_get_impl_type( config_node ) == GEN_DATA) {
	  enkf_node_type * node = enkf_state_get_node( enkf_main->ensemble[0] , key);
	  enkf_fs_fread_node( fs , node , report_step , 0 , FORECAST);
	}
      }
      active_size[ikw] = __get_active_size( config_node , report_step , active_list );
      row_offset[ikw]  = current_row_offset;
      
      if ((active_size[ikw] + current_row_offset) > matrix_size) {
	/* Not enough space in A */
	if (first_kw) {
	  /* Try to grow the matrix */
	  if (!matrix_safe_resize(A , active_size[ikw] , ens_size , false)) 
	    util_exit("%s: sorry failed to allocate %d doubles for the inner enkf update. Need more memory \n",__func__ , active_size[ikw] * ens_size);
	  matrix_size = active_size[ikw];
	} else
	  /* Do not try to grow the matrix unless we are at the first kw. */
	  add_more_kw = false;
      }

      if (add_more_kw) {
	if (active_size[ikw] > 0) {
	  state_enum load_state;
	  
	  if (hash_inc_counter( use_count , key) == 0)
	    load_state = FORECAST;   /* This is the first time this keyword is updated for this reportstep */
	  else
	    load_state = ANALYZED;

	  /** This could be multi-threaded */
	  for (int iens = 0; iens < ens_size; iens++) {
	    enkf_node_type * node = enkf_state_get_node( enkf_main->ensemble[iens] , key);
	    enkf_fs_fread_node( fs , node , report_step , iens , load_state);
	    enkf_node_matrix_serialize( node , active_list , A , row_offset[ikw] , iens);
	  }
	  current_row_offset += active_size[ikw];

	}
	
	ikw++;
	if (ikw == num_kw)
	  add_more_kw = false;
      }
      //add_more_kw = false;   /* If this is here unconditionally we will only have one node for each matrix A */
      first_kw = false;
      {
	char * label = util_alloc_sprintf("serializing: %s" , key);
	msg_update( msg , label);
	free(label);
      }
    } while (add_more_kw);
    ikw2 = ikw;
    matrix_shrink_header( A , current_row_offset , ens_size );
    if (current_row_offset > 0) {
      /* The actual update */

      msg_update(msg , " matrix multiplication");
      matrix_inplace_matmul_mt( A , X5 , 4 );  /* Four CPU threads - nothing like a little hardcoding ... */
      /* Deserialize */
      {
	for (int i = ikw1; i < ikw2; i++) {
          if (active_size[i] > 0) {
            const char             * key              = stringlist_iget(update_keys , i);
            const active_list_type * active_list      = local_ministep_get_node_active_list( ministep , key );
            {
              char * label = util_alloc_sprintf("deserializing: %s" , key);
              msg_update( msg , label);
              free(label);
            }
            
            for (int iens = 0; iens < ens_size; iens++) {
              enkf_node_type * node = enkf_state_get_node( enkf_main->ensemble[iens] , key);
              enkf_node_matrix_deserialize(node , active_list , A , row_offset[i] , iens);
              enkf_fs_fwrite_node( fs , node , report_step , iens , ANALYZED);
            }
          }
	}
      }
    }
    
    if (ikw2 == num_kw)
      complete = true;
  } while ( !complete );

  free(active_size);
  free(row_offset);
  msg_free( msg , true );
  matrix_free( A );
}




/**
   This is  T H E  EnKF update routine. 
*/


void enkf_main_UPDATE(enkf_main_type * enkf_main , int step1 , int step2) {
  /* 
     If include_internal_observations is true all observations in the
     time interval [step1+1,step2] will be used, otherwise only the
     last observation at step2 will be used.
  */
  bool include_internal_observations = analysis_config_merge_observations( enkf_main->analysis_config );
  double alpha                       = analysis_config_get_alpha( enkf_main->analysis_config ); 
  double std_cutoff                  = analysis_config_get_std_cutoff( enkf_main->analysis_config );
  const int ens_size                 = ensemble_config_get_size(enkf_main->ensemble_config);
  int start_step , end_step;
  
  /* Observe that end_step is inclusive. */
  if (include_internal_observations) {
    start_step = step1;
    end_step   = step2;
  } else {
    start_step = step2;
    end_step   = step2;
  }
  
  
  {
    /*
      Observations and measurements are collected in these temporary
      structures. obs_data is a precursor for the 'd' vector, and
      meas_forecast is a precursor for the 'S' matrix'.
      
      The reason for gong via these temporary structures is to support
      deactivating observations which should not be used in the update
      process.
    */
    
    obs_data_type     	  	* obs_data      = obs_data_alloc();
    meas_matrix_type  	  	* meas_forecast = meas_matrix_alloc( ens_size );
    meas_matrix_type  	  	* meas_analyzed = meas_matrix_alloc( ens_size );
    local_config_type 	  	* local_config  = enkf_main->local_config;
    const local_updatestep_type * updatestep    = local_config_iget_updatestep( local_config , step2 );  /* Only step2 considered */
    hash_type                   * use_count     = hash_alloc();                                          
    matrix_type                 * randrot       = NULL;
    const char                  * log_path      = analysis_config_get_log_path( enkf_main->analysis_config );
    FILE                        * log_stream;
    char                        * log_file; 


    if (analysis_config_get_random_rotation( enkf_main->analysis_config ))
      randrot = enkf_analysis_alloc_mp_randrot( ens_size ); 
    
    util_make_path( log_path );
    if (start_step == end_step)
      log_file = util_alloc_sprintf("%s%c%04d" , log_path , UTIL_PATH_SEP_CHAR , end_step);
    else
      log_file = util_alloc_sprintf("%s%c%04d-%04d" , log_path , UTIL_PATH_SEP_CHAR , start_step , end_step);
    log_stream = util_fopen( log_file , "w" );
    
    for (int ministep_nr = 0; ministep_nr < local_updatestep_get_num_ministep( updatestep ); ministep_nr++) {
      for(int report_step = start_step; report_step <= end_step; report_step++)  {
	local_ministep_type   * ministep = local_updatestep_iget_ministep( updatestep , ministep_nr );      
	
        enkf_obs_get_obs_and_measure(enkf_main->obs, enkf_main_get_fs(enkf_main), report_step, FORECAST, ens_size, 
				     (const enkf_state_type **) enkf_main->ensemble, meas_forecast, obs_data , ministep);

	meas_matrix_calculate_ens_stats( meas_forecast );
	enkf_analysis_deactivate_outliers( obs_data , meas_forecast  , std_cutoff , alpha);
        
        /* How the fuck does dup() work?? */
	enkf_analysis_fprintf_obs_summary( obs_data , meas_forecast  , report_step , local_ministep_get_name( ministep ) , stdout );
        enkf_analysis_fprintf_obs_summary( obs_data , meas_forecast  , report_step , local_ministep_get_name( ministep ) , log_stream );

	if (obs_data_get_active_size(obs_data) > 0) {
	  if (analysis_config_Xbased( enkf_main->analysis_config )) {
	    matrix_type * X = enkf_analysis_allocX( enkf_main->analysis_config , meas_forecast , obs_data , randrot);
	    
	    enkf_main_update_mulX( enkf_main , X , ministep , end_step , use_count);
	    
	    matrix_free( X );
	  }
	}
      }
    }
    fclose( log_stream );
    free( log_file );

    if (randrot != NULL)
      matrix_free( randrot );
    
    obs_data_free( obs_data );
    meas_matrix_free( meas_forecast );
    meas_matrix_free( meas_analyzed );
    enkf_main_inflate( enkf_main , step2 , use_count); 
    hash_free( use_count );
  }
}






static void enkf_main_run_wait_loop(enkf_main_type * enkf_main ) {
  const int ens_size            = ensemble_config_get_size(enkf_main->ensemble_config);
  arg_pack_type ** arg_list     = util_malloc( ens_size * sizeof * arg_list , __func__);
  thread_pool_type * tp         = thread_pool_alloc( 10 );
  job_status_type * status_list = util_malloc( ens_size * sizeof * status_list , __func__);
  const int usleep_time         = 2500000; //100000; /* 1/10 of a second */ 
  int jobs_remaining;
  int iens;
  
  for (iens = 0; iens < ens_size; iens++) {
    arg_list[iens]    = arg_pack_alloc();
    status_list[iens] = JOB_QUEUE_NOT_ACTIVE;
  }

  do {
    job_status_type status;
    jobs_remaining = 0;
    
    for (iens = 0; iens < ens_size; iens++) {
      enkf_state_type * enkf_state = enkf_main->ensemble[iens];
      status = enkf_state_get_run_status( enkf_state );
      if ((status != JOB_QUEUE_NOT_ACTIVE) && ( status != JOB_QUEUE_ALL_OK) && (status != JOB_QUEUE_ALL_FAIL))
        jobs_remaining += 1; /* OK - the job is still running. */
      
      if ((status == JOB_QUEUE_RUN_OK) || (status == JOB_QUEUE_RUN_FAIL)) {
        if (status_list[iens] != status) {
          arg_pack_append_ptr( arg_list[iens] , enkf_state );
          arg_pack_append_int( arg_list[iens] , status );
          
          thread_pool_add_job( tp , enkf_state_complete_forward_model__ , arg_list[iens] );
        }
      }
      status_list[iens] = status;
    }
    if (jobs_remaining > 0)
      usleep( usleep_time );
  } while (jobs_remaining > 0);
  thread_pool_join( tp );
  thread_pool_free( tp );

  for (iens = 0; iens < ens_size; iens++) 
    arg_pack_free( arg_list[iens] );
  free( arg_list );
  free( status_list );
}






static void enkf_main_run_step(enkf_main_type * enkf_main      , 
                               run_mode_type    run_mode       , 
                               const bool * iactive            , 
                               int load_start                  ,  /* For internalizing results. */
                               int init_step_parameter         ,     
                               state_enum init_state_parameter , 
                               state_enum init_state_dynamic   , 
                               int step1                       , 
                               int step2                       ,  /* Discarded for predictions */
                               bool enkf_update                ,     
                               forward_model_type * forward_model) {  /* The forward model will be != NULL ONLY if it is different from the default forward model. */
  
  {
    const ecl_config_type * ecl_config = enkf_main_get_ecl_config( enkf_main );
    if ((step1 > 0) && (!ecl_config_can_restart(ecl_config))) {
      fprintf(stderr,"** Warning - tried to restart case which is not properly set up for restart.\n");
      fprintf(stderr,"** Need <INIT> in datafile and INIT_SECTION keyword in config file.\n");
      util_exit("%s: exiting \n",__func__);
    }
  }
  
  {
    bool resample_when_fail  = model_config_resample_when_fail(enkf_main->model_config);
    int  max_internal_submit = model_config_get_max_internal_submit(enkf_main->model_config);
    const int ens_size       = ensemble_config_get_size(enkf_main->ensemble_config);
    int   job_size;
    int iens;
  
    if (run_mode == ENSEMBLE_PREDICTION)
      printf("Starting predictions from step: %d\n",step1);
    else
      printf("Starting forward step: %d -> %d\n",step1 , step2);
    
    log_add_message(enkf_main->logh , 1 , NULL , "===================================================================", false);
    log_add_fmt_message(enkf_main->logh , 1 , NULL , "Forward model: %d -> %d ",step1,step2);
    job_size = 0;
    for (iens = 0; iens < ens_size; iens++)
      if (iactive[iens]) job_size++;
    
    {
      pthread_t          queue_thread;
      job_queue_type * job_queue = site_config_get_job_queue(enkf_main->site_config);
      bool             verbose   = true;  
      arg_pack_type * queue_args = arg_pack_alloc();
      arg_pack_append_ptr(queue_args  , job_queue);
      arg_pack_append_int(queue_args  , job_size);
      arg_pack_append_bool(queue_args , verbose);
      arg_pack_lock( queue_args );
      pthread_create( &queue_thread , NULL , job_queue_run_jobs__ , queue_args);
      
      {
        thread_pool_type * submit_threads = thread_pool_alloc(4);
        for (iens = 0; iens < ens_size; iens++) {
          if (iactive[iens]) {
            int load_start = step1;
            if (step1 > 0)
              load_start++;
            
            enkf_state_init_run(enkf_main->ensemble[iens] , 
                                run_mode , 
                                iactive[iens] , 
                                resample_when_fail  ,
                                max_internal_submit ,
                                init_step_parameter , 
                                init_state_parameter,
                                init_state_dynamic  , 
                                load_start , 
                                step1 , 
                                step2 , 
                                forward_model);
            
            thread_pool_add_job(submit_threads , enkf_state_start_forward_model__ , enkf_main->ensemble[iens]);
          } else
            enkf_state_set_inactive( enkf_main->ensemble[iens] );
        }
        thread_pool_join(submit_threads);  /* OK: All directories for ECLIPSE simulations are ready. */
        thread_pool_free(submit_threads);
      }
      log_add_message(enkf_main->logh , 1 , NULL , "All jobs ready for running - waiting for completion" ,  false);
      enkf_main_run_wait_loop( enkf_main );
      job_queue_finalize(job_queue);             /* Must *NOT* be called before all jobs are done. */               
      arg_pack_free( queue_args );
    }
    
    
    
    {
      bool runOK   = true;  /* The runOK checks both that the external jobs have completed OK, and that the ert layer has loaded all data. */
      
      for (iens = 0; iens < ens_size; iens++) {
        if (! enkf_state_runOK(enkf_main->ensemble[iens])) {
          if ( runOK ) {
            log_add_fmt_message( enkf_main->logh , 1 , stderr , "Some models failed to integrate from DATES %d -> %d:",step1 , step2);
            runOK = false;
          }
          log_add_fmt_message( enkf_main->logh , 1 , stderr , "** Error in: %s " , enkf_state_get_run_path(enkf_main->ensemble[iens]));
        }
      }
      if (!runOK) 
        util_exit("The integration failed - check your forward model ...\n");
    }
    log_add_fmt_message(enkf_main->logh , 1 , NULL , "All jobs complete and data loaded for step: ->%d" , step2);
    
    if (enkf_update)
      enkf_main_UPDATE(enkf_main , load_start , step2);
    
    enkf_fs_fsync( enkf_main->dbase );
    printf("%s: ferdig med step: %d \n" , __func__,step2);
  }
}




void * enkf_main_get_enkf_config_node_type(const ensemble_config_type * ensemble_config, const char * key){
  enkf_config_node_type * config_node_type = ensemble_config_get_node(ensemble_config, key);
  return enkf_config_node_get_ref(config_node_type);
}

/**
   This function will initialize the necessary enkf_main structures
   before a run. Currently this means:

     1. Set the enkf_sched instance - either by loading from file or
        by using the default.
	 
     2. Set up the configuration of what should be internalized.

*/


void enkf_main_init_run( enkf_main_type * enkf_main, run_mode_type run_mode) {
  const ext_joblist_type * joblist = site_config_get_installed_jobs( enkf_main->site_config);

  model_config_set_enkf_sched( enkf_main->model_config , joblist , run_mode , site_config_get_statoil_mode( enkf_main->site_config ));
  enkf_main_init_internalization(enkf_main , run_mode);

}


/**
   The main RUN function - will run both enkf assimilations and experiments.
*/
void enkf_main_run(enkf_main_type * enkf_main            , 
		   run_mode_type    run_mode             , 
		   const bool     * iactive              ,          
		   int              init_step_parameters ,
		   int              start_report         , 
		   state_enum       start_state) {
  
  bool rerun       = analysis_config_get_rerun( enkf_main->analysis_config );
  int  rerun_start = analysis_config_get_rerun_start( enkf_main->analysis_config );

  enkf_main_init_run( enkf_main , run_mode);  
  {
    enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
    if (run_mode == ENKF_ASSIMILATION) {
      if (enkf_fs_rw_equal(fs)) {
	bool analyzed_start = false;
	bool prev_enkf_on;
	const enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
	const int num_nodes                = enkf_sched_get_num_nodes(enkf_sched);
	const int start_inode              = enkf_sched_get_node_index(enkf_sched , start_report);
	int inode;
	
	if (start_state == ANALYZED)
	  analyzed_start = true;
	else if (start_state == FORECAST)
	  analyzed_start = false;
	else
	  util_abort("%s: internal error - start_state must be analyzed | forecast \n",__func__);
	
	prev_enkf_on = analyzed_start;
	for (inode = start_inode; inode < num_nodes; inode++) {
	  const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
	  state_enum init_state_parameter;
          state_enum init_state_dynamic;
          int      init_step_parameter;
          int      load_start;
	  int 	   report_step1;
	  int 	   report_step2;
	  bool enkf_on;
	  forward_model_type * forward_model;
	  
	  enkf_sched_node_get_data(node , &report_step1 , &report_step2 , &enkf_on , &forward_model);
	  if (inode == start_inode) 
	    report_step1 = start_report;  /* If we are restarting from somewhere. */
	  
          if (rerun) {
            /* rerun ... */
            load_start           = report_step1;    /* +1 below. Observe that report_step is set to rerun_start below. */
            init_step_parameter  = report_step1;
            init_state_dynamic   = FORECAST;
            init_state_parameter = ANALYZED;
            report_step1         = rerun_start;
          } else {
            if (prev_enkf_on)
              init_state_dynamic = ANALYZED;
            else
              init_state_dynamic = FORECAST;
            /* 
               This is not a rerun - and then parameters and dynamic
               data should be initialized from the same report step.
            */
            init_step_parameter  = report_step1;
            init_state_parameter = init_state_dynamic;
            load_start = report_step1;
          }
          
          if (load_start > 0)
            load_start++;

	  enkf_main_run_step(enkf_main , ENKF_ASSIMILATION , iactive , load_start , init_step_parameter , init_state_parameter , init_state_dynamic , report_step1 , report_step2 , enkf_on , forward_model);
	  prev_enkf_on = enkf_on;
	}
      } else
	fprintf(stderr , "\n** Error: when running EnKF read and write directories must be equal.\n\n");
    } else {
      /* It is an experiment */
      const enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
      const int last_report              = enkf_sched_get_last_report(enkf_sched);
      if (run_mode == ENSEMBLE_EXPERIMENT || run_mode == ENSEMBLE_PREDICTION) {
	/* No possibility to use funky forward model */
        int load_start = start_report;
        state_enum init_state_parameter = start_state;
        state_enum init_state_dynamic   = start_state;
	enkf_main_run_step(enkf_main , run_mode , iactive , load_start , init_step_parameters , init_state_parameter , init_state_dynamic , start_report , last_report , false , NULL );
      } 
    }
  }
}



void enkf_main_initialize(enkf_main_type * enkf_main , const stringlist_type * param_list , int iens1 , int iens2) {
  int iens;
  msg_type * msg = msg_alloc("Initializing...: " );
  msg_show(msg);

  for (iens = iens1; iens <= iens2; iens++) {
    enkf_state_type * state = enkf_main_iget_state( enkf_main , iens);
    {
      char * iens_string = util_alloc_sprintf("%04d" , iens);
      msg_update(msg , iens_string); 
      free(iens_string);
    }
    enkf_state_initialize( state , param_list );
    msg_update(msg , "Done");
  }
  msg_free(msg , false);
}


/**
   This function creates a local_config file corresponding to the
   default 'ALL_ACTIVE' configuration. We eat our own shit around here ... 
*/
   
void enkf_main_create_all_active_config( const enkf_main_type * enkf_main , const char * local_config_file ) {
  FILE * stream = util_fopen( local_config_file , "w");

  fprintf(stream , "%-32s ALL_ACTIVE\n", local_config_get_cmd_string( CREATE_UPDATESTEP ));
  fprintf(stream , "%-32s ALL_ACTIVE\n", local_config_get_cmd_string( CREATE_MINISTEP ));
  fprintf(stream , "%-32s ALL_ACTIVE ALL_ACTIVE\n" , local_config_get_cmd_string( ATTACH_MINISTEP ));
  
  /* Adding all observation keys */
  {
    hash_iter_type * obs_iter = enkf_obs_alloc_iter( enkf_main->obs );
    while ( !hash_iter_is_complete(obs_iter) ) {
      const char * obs_key = hash_iter_get_next_key( obs_iter );
      fprintf(stream , "%-32s ALL_ACTIVE %s\n",local_config_get_cmd_string( ADD_OBS ) , obs_key);
    }
    hash_iter_free( obs_iter );
  }
  
  /* Adding all node which can be updated. */
  {
    stringlist_type * keylist = ensemble_config_alloc_keylist_from_var_type( enkf_main->ensemble_config , PARAMETER + DYNAMIC_STATE + DYNAMIC_RESULT);
    int i;
    for (i = 0; i < stringlist_get_size( keylist ); i++) {
      const char * key = stringlist_iget( keylist , i);
      const enkf_config_node_type * config_node = ensemble_config_get_node( enkf_main->ensemble_config , key );
      bool add_node = true;
      
      /* 
         We make sure that summary nodes which are not observed
         are not updated. I.e. the total production in a
         well. 

         This was changed by request ... 
      */


      //if (enkf_config_node_get_var_type( config_node ) == DYNAMIC_RESULT)
      //  if (enkf_config_node_get_num_obs( config_node ) == 0)
      //    add_node = false;
      
      /*
        Make sure the funny GEN_KW instance masquerading as
        SCHEDULE_PREDICTION_FILE is not added to the soup.
      */
      if (util_string_equal(key , "PRED"))
        add_node = false;
      
      if (add_node)
        fprintf(stream , "%-32s ALL_ACTIVE %s\n",local_config_get_cmd_string( ADD_DATA ) , key);
    }
    stringlist_free( keylist);
  }
  
  /* Install the ALL_ACTIVE step as the default. */
  fprintf(stream , "%-32s ALL_ACTIVE" , local_config_get_cmd_string( INSTALL_DEFAULT_UPDATESTEP ));
  fclose( stream );
}





static config_type * enkf_main_alloc_config() {
  config_type * config = config_alloc();
  config_item_type * item;
    
  /*****************************************************************/
  /* config_add_item():                                            */
  /*                                                               */
  /*  1. boolean - required?                                       */
  /*  2. boolean - append?                                         */
  /*****************************************************************/
  
  /*****************************************************************/
  /** Keywords expected normally found in site_config */
  item = config_add_item(config , "HOST_TYPE" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  config_item_set_common_selection_set(item , 2, (const char *[2]) {"STATOIL" , "HYDRO"});
  config_set_arg( config , "HOST_TYPE" , 1 , (const char *[1]) { DEFAULT_HOST_TYPE });
  
  item = config_add_item(config , "CASE_TABLE" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
  
  config_add_key_value( config , "LOG_LEVEL" , false , CONFIG_INT);
  config_add_key_value( config , "LOG_FILE"  , false , CONFIG_STRING);
  
  
  item = config_add_item(config , "MAX_SUBMIT" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
  config_set_arg(config , "MAX_SUBMIT" , 1 , (const char *[1]) { DEFAULT_MAX_SUBMIT});
  
  config_add_key_value(config , "RESAMPLE_WHEN_FAIL" , false , CONFIG_BOOLEAN);
  config_add_key_value(config , "MAX_RETRY" , false , CONFIG_INT);
      

  item = config_add_item(config , "QUEUE_SYSTEM" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  {
    stringlist_type * lsf_dep    = stringlist_alloc_argv_ref( (const char *[2]) {"LSF_QUEUE" , "MAX_RUNNING_LSF"}   , 2);
    stringlist_type * rsh_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"RSH_HOST_LIST" , "RSH_COMMAND" , "MAX_RUNNING_RSH"} , 2);
    stringlist_type * local_dep  = stringlist_alloc_argv_ref( (const char *[1]) {"MAX_RUNNING_LOCAL"}   , 1);
    
    config_item_set_common_selection_set( item , 3 , (const char *[3]) {"LSF" , "LOCAL" , "RSH"});
    config_item_set_required_children_on_value( item , "LSF"   , lsf_dep);
    config_item_set_required_children_on_value( item , "RSH"   , rsh_dep);
    config_item_set_required_children_on_value( item , "LOCAL" , local_dep);
    
    stringlist_free(lsf_dep);
    stringlist_free(rsh_dep);
    stringlist_free(local_dep);
  }


  /* 
     You can set environment variables which will be applied to the
     run-time environment. Can unfortunately not use constructions
     like PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
  */
  item = config_add_item(config , "SETENV" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);

  /**
     UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib
     
     Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
  */
  item = config_add_item(config , "UPDATE_PATH" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);

  item = config_add_item( config , "LICENSE_PATH" , true , false );
  config_item_set_argc_minmax(item , 1 , 1, NULL );
  
    
  /*****************************************************************/
  /* Items related to running jobs with lsf/rsh/local ...          */
  
  /* These must be set IFF QUEUE_SYSTEM == LSF */
  item = config_add_item(config , "LSF_QUEUE"     , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  
  item = config_add_item(config , "MAX_RUNNING_LSF" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


  /* These must be set IFF QUEUE_SYSTEM == RSH */
  config_add_item(config , "RSH_HOST_LIST" , false , false);
  item = config_add_item(config , "RSH_COMMAND" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXECUTABLE});
  item = config_add_item(config , "MAX_RUNNING_RSH" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
  
    
  /* These must be set IFF QUEUE_SYSTEM == LOCAL */
  item = config_add_item(config , "MAX_RUNNING_LOCAL" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});

    
  item = config_add_item(config , "JOB_SCRIPT" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

  item = config_add_item(config , "INSTALL_JOB" , true , true);
  config_item_set_argc_minmax(item , 2 , 2 , (const config_item_types [2]) {CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  
  /* Plotting stuff */
  item = config_add_key_value(config , "IMAGE_TYPE" , false , CONFIG_STRING);
  config_item_set_common_selection_set( item , 3 , (const char *[3]) {"png" , "jpg" , "psc"});
  
  item = config_add_key_value(config , "PLOT_DRIVER" , false , CONFIG_STRING);
  config_item_set_common_selection_set( item , 2 , (const char *[2]) {"PLPLOT" , "TEXT"});
  

  plot_config_add_config_items( config );
  
    
  /*****************************************************************/
  /* Required keywords from the ordinary model_config file */
  item = config_add_item(config , "NUM_REALIZATIONS" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
  config_add_alias(config , "NUM_REALIZATIONS" , "SIZE");
  config_add_alias(config , "NUM_REALIZATIONS" , "NUM_REALISATIONS");
  config_install_message(config , "SIZE" , "** Warning: \'SIZE\' is depreceated - use \'NUM_REALIZATIONS\' instead.");
  

  item = config_add_item(config , "GRID" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "ECLBASE" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  
  item = config_add_item(config , "SCHEDULE_FILE" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
  /* 
     Observe that SCHEDULE_PREDICTION_FILE - which is implemented as a GEN_KW is
     added en ensemble_config.c 
  */

  item = config_add_item(config , "DATA_FILE" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

  item = config_add_item(config , "INIT_SECTION" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_FILE});
  config_add_alias(config , "INIT_SECTION" , "EQUIL_INIT_FILE"); 
  
  /*****************************************************************/
  /* Optional keywords from the model config file */
  
  item = config_add_item( config , "RUN_TEMPLATE" , false , true );
  config_item_set_argc_minmax(item , 2 , -1 , (const config_item_types [2]) { CONFIG_EXISTING_FILE , CONFIG_STRING });  /* Force the template to exist at boot time. */

  config_add_key_value(config , "RUNPATH" , false , CONFIG_STRING);
  
  item = config_add_item(config , "ENSPATH" , true , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  config_set_arg(config , "ENSPATH" , 1 , (const char *[1]) { DEFAULT_ENSPATH });
  
  item = config_add_item(config , "DBASE_TYPE" , true , false);
  config_item_set_argc_minmax(item , 1, 1 , NULL);
  config_item_set_common_selection_set(item , 3 , (const char *[3]) {"PLAIN" , "SQLITE" , "BLOCK_FS"});
  config_set_arg(config , "DBASE_TYPE" , 1 , (const char *[1] ) { DEFAULT_DBASE_TYPE });
  
  item = config_add_item(config , "FORWARD_MODEL" , true , true);
  config_item_set_argc_minmax(item , 1 , -1 , NULL);
  
  item = config_add_item(config , "DATA_KW" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);
  
  item = config_add_item(config , "KEEP_RUNPATH" , false , false);
  config_item_set_argc_minmax(item , 1 , -1 , NULL);
  
  item = config_add_item(config , "DELETE_RUNPATH" , false , false);
  config_item_set_argc_minmax(item , 1 , -1 , NULL);

  item = config_add_item(config , "ADD_STATIC_KW" , false , true);
  config_item_set_argc_minmax(item , 1 , -1 , NULL);
  
  item = config_add_item(config , "ADD_FIXED_LENGTH_SCHEDULE_KW" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , (const config_item_types [2]) { CONFIG_STRING , CONFIG_INT});
    
  item = config_add_item(config , "OBS_CONFIG"  , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

  item = config_add_item(config , "LOCAL_CONFIG"  , false , true);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "REFCASE" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

  item = config_add_item(config , "ENKF_SCHED_FILE" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

  item = config_add_item(config , "HISTORY_SOURCE" , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  {
    stringlist_type * refcase_dep = stringlist_alloc_argv_ref( (const char *[1]) {"REFCASE"} , 1);
    
    config_item_set_common_selection_set(item , 3 , (const char *[3]) {"SCHEDULE" , "REFCASE_SIMULATED" , "REFCASE_HISTORY"});
    config_item_set_required_children_on_value(item , "REFCASE_SIMULATED" , refcase_dep);
    config_item_set_required_children_on_value(item , "REFCASE_HISTORY"  , refcase_dep);

    stringlist_free(refcase_dep);
  }
  config_set_arg(config , "HISTORY_SOURCE" , 1 , (const char *[1]) { DEFAULT_HISTORY_SOURCE });
  
  
  /*****************************************************************/
  /* 
     Keywords for the analysis - all optional. The analysis_config object
     is instantiated with defaults from enkf_defaults.h
  */
  item = config_add_key_value(config , "ENKF_MODE" , false , CONFIG_STRING );
  config_item_set_common_selection_set(item , 2 , (const char *[2]) {"STANDARD" , "SQRT"});
  config_add_key_value( config , "ENKF_TRUNCATION" , false , CONFIG_FLOAT);
  config_add_key_value( config , "ENKF_ALPHA" , false , CONFIG_FLOAT);
  config_add_key_value( config , "ENKF_MERGE_OBSERVATIONS" , false , CONFIG_BOOLEAN);
  config_add_key_value( config , "ENKF_RERUN" , false , CONFIG_BOOLEAN);
  config_add_key_value( config , "RERUN_START" , false , CONFIG_INT);
  
  /*****************************************************************/
  /* Keywords for the estimation                                   */
  ensemble_config_add_config_items(config); 
  return config;
}


static void enkf_main_iset_keep_runpath( enkf_main_type * enkf_main , int index , keep_runpath_type keep) {
  enkf_main->keep_runpath[index] = keep;
}


void enkf_main_parse_keep_runpath(enkf_main_type * enkf_main , const char * keep_runpath_string , const char * delete_runpath_string) {
  const int ens_size = ensemble_config_get_size( enkf_main->ensemble_config );
  
  int i;
  for (i = 0; i < ens_size; i++) 
  enkf_main_iset_keep_runpath( enkf_main , i , DEFAULT_KEEP);
  
  {
    bool * flag = util_malloc( sizeof * flag * ens_size , __func__);
    
    util_sscanf_active_range(keep_runpath_string , ens_size - 1 , flag);
    for (i = 0; i < ens_size; i++) {
      if (flag[i]) 
        enkf_main_iset_keep_runpath( enkf_main , i , EXPLICIT_KEEP);
    }
    
    free( flag );
  }
  
  {
    bool * flag = util_malloc( sizeof * flag * ens_size , __func__);
    
    util_sscanf_active_range(delete_runpath_string , ens_size - 1 , flag);
    for (i = 0; i < ens_size; i++) {
      if (flag[i]) {
        if (enkf_main->keep_runpath[i] == EXPLICIT_KEEP)
          util_abort("%s: Inconsistent use of KEEP_RUNPATH / DELETE_RUNPATH - trying to both keep and delete member:%d \n",__func__ , i);
        enkf_main_iset_keep_runpath( enkf_main , i , EXPLICIT_DELETE);
      }
    }
    
    free(flag );
  }
}


static enkf_main_type * enkf_main_alloc_empty(hash_type * config_data_kw) {
  enkf_main_type * enkf_main = util_malloc(sizeof *enkf_main, __func__);
  UTIL_TYPE_ID_INIT(enkf_main , ENKF_MAIN_ID);  
  enkf_main->dbase        = NULL;
  enkf_main->ensemble     = NULL;
  enkf_main->keep_runpath = NULL;

  
  /* Here we add the functions which should be available for string substitution operations. */
  enkf_main->subst_func_pool = subst_func_pool_alloc( );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "EXP"       , "exp"                              , subst_func_exp         , false , 1 , 1 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "LOG"       , "log"                              , subst_func_log         , false , 1 , 1 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "POW10"     , "Calculates 10^x"                  , subst_func_pow10       , false , 1 , 1 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "ADD"       , "Adds arguments"                   , subst_func_add         , true  , 1 , 0 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "MUL"       , "Multiplies arguments"             , subst_func_mul         , true  , 1 , 0 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "RANDINT"   , "Returns a random integer"         , subst_func_randint     , false , 0 , 0 );
  subst_func_pool_add_func( enkf_main->subst_func_pool , "RANDFLOAT" , "Returns a random float : %12.10f" , subst_func_randfloat   , false , 0 , 0 );
  
  /**
     Allocating the parent subst_list instance. This will (should ...)
     be the top level subst instance for all substitions in the ert
     program.
     
     All the functions available or only installed in this
     subst_list.
     
     The key->value replacements installed in this instance are
     key,value pairs which are:

      o Common to all ensemble members.
       
      o Constant in time.

  */

  enkf_main->subst_list = subst_list_alloc( enkf_main->subst_func_pool );
  /* Installing the functions. */
  subst_list_insert_func( enkf_main->subst_list , "EXP"         , "__EXP__");
  subst_list_insert_func( enkf_main->subst_list , "LOG"         , "__LOG__");
  subst_list_insert_func( enkf_main->subst_list , "POW10"       , "__POW10__");
  subst_list_insert_func( enkf_main->subst_list , "ADD"         , "__ADD__");
  subst_list_insert_func( enkf_main->subst_list , "MUL"         , "__MUL__");
  subst_list_insert_func( enkf_main->subst_list , "RANDINT"     , "__RANDINT__");
  subst_list_insert_func( enkf_main->subst_list , "RANDFLOAT"   , "__RANDFLOAT__");
  

  /*
    Installing the DATA_KW keywords supplied by the user - these are
    at the very top level, so they can reuse everything defined later.
  */
  
  {
    hash_iter_type * iter = hash_iter_alloc(config_data_kw);
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      char * tagged_key = enkf_util_alloc_tagged_string( key );
      subst_list_insert_copy( enkf_main->subst_list , tagged_key , hash_get( config_data_kw , key ) , "Supplied by the user in the configuration file.");
      key = hash_iter_get_next_key(iter);
      free( tagged_key );
    }
    hash_iter_free(iter);
  }
  
  
  
  /* 
     Installing the based (key,value) pairs which are common to all
     ensemble members, and independent of time.
  */
  {
    char * cwd             = util_alloc_cwd();
    char * date_string     = util_alloc_date_stamp();

    char * cwd_key         = enkf_util_alloc_tagged_string( "CWD" );   
    char * config_path_key = enkf_util_alloc_tagged_string( "CONFIG_PATH" );   
    char * date_key        = enkf_util_alloc_tagged_string( "DATE" );   

    subst_list_insert_owned_ref( enkf_main->subst_list , cwd_key         , cwd , "The current working directory we are running from - the location of the config file.");
    subst_list_insert_ref( enkf_main->subst_list , config_path_key , cwd , "The current working directory we are running from - the location of the config file.");
    subst_list_insert_owned_ref( enkf_main->subst_list , date_key        , date_string , "The current date");
    
    free( cwd_key );
    free( config_path_key );
    free( date_key );
  }
  enkf_main->templates    = ert_templates_alloc( enkf_main->subst_list );
  return enkf_main;
}
  



static void enkf_main_alloc_members( enkf_main_type * enkf_main , hash_type * data_kw) {
  stringlist_type * keylist  = ensemble_config_alloc_keylist(enkf_main->ensemble_config);
  int ens_size               = ensemble_config_get_size( enkf_main->ensemble_config );
  
  int keys        = stringlist_get_size(keylist);
  msg_type * msg  = msg_alloc("Initializing member: ");
  msg_show(msg);
  
  enkf_main->ensemble = util_malloc(ensemble_config_get_size(enkf_main->ensemble_config) * sizeof * enkf_main->ensemble , __func__);
  for (int iens = 0; iens < ens_size; iens++) {
    msg_update_int(msg , "%03d" , iens);
    enkf_main->ensemble[iens] = enkf_state_alloc(iens,
                                                 enkf_main->dbase , 
                                                 model_config_iget_casename( enkf_main->model_config , iens ) , 
                                                 enkf_main->keep_runpath[iens]                                ,
                                                 enkf_main->model_config                                      , 
                                                 enkf_main->ensemble_config                                   ,
                                                 enkf_main->site_config                                       , 
                                                 enkf_main->ecl_config                                        ,
                                                 model_config_get_std_forward_model(enkf_main->model_config),
                                                 enkf_main->logh,
                                                 enkf_main->templates,
                                                 enkf_main->subst_list);
  }
  msg_free(msg , true);
  
  msg  = msg_alloc("Adding key: ");
  msg_show(msg);
  for (int ik = 0; ik < keys; ik++) {
    const char * key = stringlist_iget(keylist, ik);
    msg_update(msg , key);
    const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_main->ensemble_config , key);
    for (int iens = 0; iens < ens_size; iens++) 
      enkf_state_add_node(enkf_main->ensemble[iens] , key , config_node);
  }
  msg_free(msg , true);
  stringlist_free(keylist);
}



void enkf_main_remount_fs( enkf_main_type * enkf_main ) {
  const model_config_type * model_config = enkf_main->model_config;
  const char * mount_map = "enkf_mount_info";
  enkf_main->dbase = enkf_fs_mount(model_config_get_enspath(model_config ) , model_config_get_dbase_type( model_config ) , mount_map );
}




/******************************************************************/
/* 
   Adding inverse observation keys to the enkf_nodes; can be called 
   several times.
*/


void enkf_main_update_obs_keys( enkf_main_type * enkf_main ) {
  /* First clear all existing observation keys. */
  ensemble_config_clear_obs_keys( enkf_main->ensemble_config );

  /* Add new observation keys. */
  {
    hash_type      * map  = enkf_obs_alloc_data_map(enkf_main->obs);
    hash_iter_type * iter = hash_iter_alloc(map);
    const char * obs_key  = hash_iter_get_next_key(iter);
    while (obs_key  != NULL) {
      const char * state_kw = hash_get(map , obs_key);
      ensemble_config_add_obs_key(enkf_main->ensemble_config , state_kw , obs_key);
      obs_key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    hash_free(map);
  }
}

  
/**
   This function boots everything needed for running a EnKF
   application. Very briefly it can be summarized as follows:

    1. A large config object is initalized with all the possible
       keywords we are looking for.

    2. All the config files are parsed in one go.

    3. The various objects are build up by reading from the config
       object.

    4. The resulting enkf_main object contains *EVERYTHING*
       (whoaha...)
*/



enkf_main_type * enkf_main_bootstrap(const char * _site_config, const char * _model_config) {
  const char     * site_config  = getenv("ENKF_SITE_CONFIG");
  char           * model_config;
  enkf_main_type * enkf_main;    /* The enkf_main object is allocated when the config parsing is completed. */
  
  
  if (site_config == NULL)
    site_config = _site_config;
  
  if (site_config == NULL) 
    util_exit("%s: main enkf_config file is not set. Use environment variable \"ENKF_SITE_CONFIG\" - or recompile - aborting.\n",__func__);
  printf("site config : %s \n\n",site_config);
  {
    char * path;
    char * base;
    char * ext;
    util_alloc_file_components(_model_config , &path , &base , &ext);
    if (path != NULL) {
      if (chdir(path) != 0) 
	util_abort("%s: failed to change directory to: %s : %s \n",__func__ , path , strerror(errno));

      printf("Changing to directory ...................: %s \n",path);
      if (ext != NULL) {
	model_config = util_alloc_joined_string((const char *[3]) {base , "." , ext} , 3 , "");
	free(base);
      } else 
	model_config = base;
      
      free(ext);
      free(path);
    } else
      model_config = util_alloc_string_copy(_model_config);
  }  
  
  if (!util_file_exists(site_config))  util_exit("%s: can not locate site configuration file:%s \n",__func__ , site_config);
  if (!util_file_exists(model_config)) util_exit("%s: can not locate user configuration file:%s \n",__func__ , model_config);
  {  
    config_type * config = enkf_main_alloc_config();
    config_parse(config , site_config  , "--" , "INCLUDE" , "DEFINE" , enkf_util_alloc_tagged_string , false , false);
    config_parse(config , model_config , "--" , "INCLUDE" , "DEFINE" , enkf_util_alloc_tagged_string , false , true);
    /*****************************************************************/
    /* OK - now we have parsed everything - and we are ready to start
       populating the enkf_main object. 
    */

    {
      hash_type      * data_kw   = config_alloc_hash(config , "DATA_KW");
      enkf_main = enkf_main_alloc_empty( data_kw );
      hash_free( data_kw );
    }


    /* The log object */
    {
      char * log_file = util_alloc_filename(NULL , model_config , DEFAULT_LOG_FILE);
      enkf_main->logh = log_alloc_existing( log_file , DEFAULT_LOG_LEVEL);
      free( log_file );
    }
    
    if (config_item_set( config , "LOG_LEVEL")) 
      log_set_level( enkf_main->logh , config_get_value_as_int(config , "LOG_LEVEL"));
    
    if (config_item_set( config , "LOG_FILE")) 
      log_reset_filename( enkf_main->logh , config_get_value(config , "LOG_FILE"));
      
    printf("Activity will be logged to ..............: %s \n",log_get_filename( enkf_main->logh ));
    log_add_message(enkf_main->logh , 1 , NULL , "ert configuration loaded" , false);



    /* Plot info */
    {
      enkf_main->plot_config = plot_config_alloc();
      plot_config_init_from_config( enkf_main->plot_config , config );
    }




    enkf_main->analysis_config = analysis_config_alloc( );
    analysis_config_init_from_config( enkf_main->analysis_config , config );
    {
      bool use_lsf;
      enkf_main->ecl_config      = ecl_config_alloc( config );
      enkf_main->ensemble_config = ensemble_config_alloc( config , ecl_config_get_grid( enkf_main->ecl_config ) );
      enkf_main->site_config     = site_config_alloc(config , ensemble_config_get_size( enkf_main->ensemble_config ) , &use_lsf);
      enkf_main->model_config    = model_config_alloc(config , 
                                                      ensemble_config_get_size( enkf_main->ensemble_config ),
						      site_config_get_installed_jobs(enkf_main->site_config) , 
						      ecl_config_get_last_history_restart( enkf_main->ecl_config ), 
						      ecl_config_get_sched_file(enkf_main->ecl_config) , 
						      site_config_get_statoil_mode( enkf_main->site_config ),
						      use_lsf);
    }


    /*****************************************************************/
    /**
       To keep or not to keep the runpath directories? The problem is
       that the default behavior is different depending on the run_mode:

       enkf_mode: In this case the default behaviour is to delete the
       runpath directories. You can explicitly say that you want to
       keep runpath directories with the KEEP_RUNPATH
       directive. 

       experiments: In this case the default is to keep the runpath
       directories around, but you can explicitly say that you
       want to remove the directories by using the DELETE_RUNPATH
       option.

       The final decision is performed in enkf_state().
    */
    {

      {
        char * keep_runpath_string   = NULL;
        char * delete_runpath_string = NULL;
      
        enkf_main->keep_runpath = util_malloc( sizeof * enkf_main->keep_runpath * ensemble_config_get_size( enkf_main->ensemble_config ) , __func__);
        
        if (config_has_set_item(config , "KEEP_RUNPATH")) 
          keep_runpath_string = config_alloc_joined_string(config , "KEEP_RUNPATH" , "");
        
        if (config_has_set_item(config , "DELETE_RUNPATH")) 
          delete_runpath_string = config_alloc_joined_string(config , "DELETE_RUNPATH" , "");

        enkf_main_parse_keep_runpath( enkf_main , keep_runpath_string , delete_runpath_string);
        
        util_safe_free( keep_runpath_string );
        util_safe_free( delete_runpath_string );
      }
      
      
      if (config_has_set_item(config , "ADD_STATIC_KW")) {
	for (int i=0; i < config_get_occurences(config , "ADD_STATIC_KW"); i++) {
	  const stringlist_type * static_kw_list = config_iget_stringlist_ref(config , "ADD_STATIC_KW" , i);
	  int k;
	  for (k = 0; k < stringlist_get_size(static_kw_list); k++)
	    ecl_config_add_static_kw(enkf_main->ecl_config , stringlist_iget( static_kw_list , k));
	}
      }

      /* Installing templates */
      {
        for (int i=0; i < config_get_occurences( config , "RUN_TEMPLATE"); i++) {
          const char * template_file = config_iget( config , "RUN_TEMPLATE" , i , 0);
          const char * target_file   = config_iget( config , "RUN_TEMPLATE" , i , 1);
          ert_template_type * template = ert_templates_add_template( enkf_main->templates , template_file , target_file );

          for (int iarg = 2; iarg < config_get_occurence_size( config , "RUN_TEMPLATE" , i); iarg++) {
            char * key , *value;
            util_binary_split_string( config_iget( config , "RUN_TEMPLATE" , i , iarg ), "=:" , true , &key , &value);
            
            if (value != NULL) {
              char * tagged_key = enkf_util_alloc_tagged_string( key );
              ert_template_add_arg( template , tagged_key , value );
              free( tagged_key );
            } else
              fprintf(stderr,"** Warning - failed to parse argument:%s as key:value - ignored \n",config_iget( config , "RUN_TEMPLATE" , i , iarg ));
            
            free( key );
            util_safe_free( value );
          }
        }
      }

      
      {
	const char * obs_config_file;
	if (config_has_set_item(config , "OBS_CONFIG"))
	  obs_config_file = config_iget(config  , "OBS_CONFIG" , 0,0);
	else
	  obs_config_file = NULL;
	
	enkf_main->obs = enkf_obs_fscanf_alloc(obs_config_file , model_config_get_history(enkf_main->model_config) , enkf_main->ensemble_config);
      }

      enkf_main_update_obs_keys(enkf_main);
      

      enkf_main_remount_fs( enkf_main );
      /*****************************************************************/
      /* Adding ensemble members */
      {
	hash_type       * data_kw  = config_alloc_hash(config , "DATA_KW");
	enkf_main_alloc_members( enkf_main  , data_kw);
	hash_free(data_kw);
      }

      /*****************************************************************/
      /* 
         Installing the local_config object. Observe that the
         ALL_ACTIVE local_config configuration is ALWAYS loaded. But
         if you have created a personal local config that will be
         loaded on top.
      */
      
      {
        enkf_main->local_config  = local_config_alloc( /*enkf_main->ensemble_config , enkf_main->enkf_obs , */ model_config_get_last_history_restart( enkf_main->model_config ));
        
        /* First installing the default ALL_ACTIVE configuration. */
        {
          char * all_active_config_file = util_alloc_tmp_file("/tmp" , "enkf_local_config" , true);
          enkf_main_create_all_active_config( enkf_main , all_active_config_file );
          local_config_load( enkf_main->local_config , all_active_config_file , enkf_main->logh);
          unlink( all_active_config_file );
          free(all_active_config_file);
        }
        
        /* Install custom local_config - if present.*/
        {
          int i;
          for (i = 0; i < config_get_occurences( config , "LOCAL_CONFIG"); i++) {
            const stringlist_type * files = config_iget_stringlist_ref(config , "LOCAL_CONFIG" , i);
            for (int j=0; j < stringlist_get_size( files ); j++)
              local_config_load( enkf_main->local_config , stringlist_iget( files , j), enkf_main->logh);
          }
        }
      }
    }
    config_free(config);
  }
  free(model_config);
  enkf_main->misfit_table = NULL;
  return enkf_main;
}
    

/**
   Sets the misfit_table of the enkf_main object. If a misfit table is
   already installed the currently installed misfit table is freed first. 

   The enkf_main object takes ownership of the input misfit table,
   i.e. the calling scope should not free the table.
*/

void enkf_main_set_misfit_table( enkf_main_type * enkf_main , misfit_table_type * misfit) {
  if (enkf_main->misfit_table != NULL)
    misfit_table_free( enkf_main->misfit_table );

  enkf_main->misfit_table = misfit;
}


misfit_table_type * enkf_main_get_misfit_table( const enkf_main_type * enkf_main ) {
  return enkf_main->misfit_table;
}



/**
   First deleting all the nodes - then the configuration.
*/

void enkf_main_del_node(enkf_main_type * enkf_main , const char * key) {
  const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
  int iens;
  for (iens = 0; iens < ens_size; iens++) 
    enkf_state_del_node(enkf_main->ensemble[iens] , key);
  ensemble_config_del_node(enkf_main->ensemble_config , key);
}




enkf_state_type ** enkf_main_get_ensemble( enkf_main_type * enkf_main) {
  return enkf_main->ensemble;
}



/**
   In this function we initialize the the variables which control
   which nodes are internalized (i.e. loaded from the forward
   simulation and stored in the enkf_fs 'database'). The system is
   based on two-levels:

   * Should we store the state? This is goverened by the variable
     model_config->internalize_state. If this is true we will
     internalize all nodes which have enkf_var_type = {dynamic_state ,
     static_state}. In the same way the variable
     model_config->internalize_results governs whether the dynamic
     results (i.e. summary variables in ECLIPSE speak) should be
     internalized.

   * In addition we have fine-grained control in the enkf_config_node
     objects where we can explicitly say that, altough we do not want
     to internalize the full state, we want to internalize e.g. the
     pressure field.
 
   * All decisions on internalization are based on a per report step
     basis.
   
   The user-space API for manipulating this is (extremely)
   limited. What is implemented here is the following:

     1. We internalize the initial dynamic state.

     2. For all the end-points in the current enkf_sched instance we
        internalize the state.

     3. store_results is set to true for all report steps irrespective
        of run_mode.

     4. We iterate over all the observations, and ensure that the
        observed nodes (i.e. the pressure for an RFT) are internalized
        (irrespective of whether they are of type dynamic_state or
        dynamic_result).

   Observe that this cascade can result in some nodes, i.e. a rate we
   are observing, to be marked for internalization several times -
   that is no problem.
    
   -----
   
   For performance reason model_config contains two bool vectors
   __load_state and __load_result; if they are true the state and
   summary are loaded from disk, otherwise no loading is
   performed. This implies that if we do not want to internalize the
   full state but for instance the pressure (i.e. for an RFT) we must
   set the __load_state variable for the actual report step to
   true. For this reason calls enkf_config_node_internalize() must be
   accompanied by calls to model_config_set_load_state|results() -
   this is ensured when using this function to manipulate the
   configuration of internalization.

*/


void enkf_main_init_internalization( enkf_main_type * enkf_main , run_mode_type run_mode ) {
  /* Clearing old internalize flags. */
  model_config_init_internalization( enkf_main->model_config );
  ensemble_config_init_internalization( enkf_main->ensemble_config );
  
  /* Internalizing the initial state. */
  model_config_set_internalize_state( enkf_main->model_config , 0);
  
  /* We internalize all the endpoints in the enkf_sched. */
  {
    int inode;
    enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
    for (inode = 0; inode < enkf_sched_get_num_nodes( enkf_sched ); inode++) {
      const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
      int report_step2            = enkf_sched_node_get_last_step( node );
      model_config_set_internalize_state( enkf_main->model_config , report_step2);
    }
  }


  /* Make sure we internalize at all observation times.*/
  {
    hash_type      * map  = enkf_obs_alloc_data_map(enkf_main->obs);
    hash_iter_type * iter = hash_iter_alloc(map); 
    const char * obs_key  = hash_iter_get_next_key(iter);
    
    while (obs_key != NULL) {
      obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_main->obs , obs_key );
      enkf_config_node_type * data_node = obs_vector_get_config_node( obs_vector );
      int active_step = -1;
      do {
	active_step = obs_vector_get_next_active_step( obs_vector , active_step );
	if (active_step >= 0) {
	  enkf_config_node_set_internalize( data_node , active_step );
	  {
	    enkf_var_type var_type = enkf_config_node_get_var_type( data_node );
	    if (var_type == DYNAMIC_STATE) 
	      model_config_set_load_state( enkf_main->model_config , active_step);
	  }
	}
      } while (active_step >= 0);
      obs_key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    hash_free(map);
  }
}
  



/*****************************************************************/

/**
   This function stores a pid file for the running ert instance. The
   rules of the game are as follows:

     1. The name of the file is just the pid number.

     2. The content of the file is the current executable and the uid
        of the current user, separated with a space.
        
     3. The argument to the function is argv[0] from the main
        function.

        
   On normal exit the file is removed with the enkf_main_delete_pid()
   function.
*/


void enkf_main_store_pid(const char * argv0) {
  const mode_t var_dir_mode = S_IRUSR + S_IWUSR + S_IXUSR + S_IRGRP + S_IWGRP + S_IXGRP + S_IROTH + S_IXOTH + S_IWOTH; /* = a+rwx */
  char * current_executable;
  if (util_is_abs_path( argv0 ))
    current_executable = util_alloc_string_copy( argv0 );
  else 
    current_executable = util_alloc_PATH_executable( argv0 );

  if (util_make_path2( DEFAULT_VAR_DIR , var_dir_mode )) {
    char * pidfile = util_alloc_sprintf("%s/%d" , DEFAULT_VAR_DIR , getpid());
    FILE * stream  = util_fopen( pidfile , "w");
    
    fprintf(stream , "%s %d\n", current_executable , getuid());
    fclose( stream );
    util_chmod_if_owner( pidfile , S_IRUSR + S_IWUSR + S_IRGRP + S_IWGRP + S_IROTH + S_IWOTH); /* chmod a+rw */
    free( pidfile );
  } else
    fprintf(stderr,"** Failed to make directory:%s \n",DEFAULT_VAR_DIR);
  
  util_safe_free( current_executable );
}



/**
   This function is called when the ert application is exiting, it
   will just delete the pid file.
   
   If the ert application exits through a crash the pid file will be
   left hanging around. In that case it will be deleted the
   next time the enkf_main_list_users() function is run.
*/

void enkf_main_delete_pid( ) {
  char * pidfile = util_alloc_sprintf("%s/%d" , DEFAULT_VAR_DIR , getpid());
  util_unlink_existing( pidfile );
  free( pidfile );
}




/* Used by external application - this is a library ... */
void  enkf_main_list_users(  set_type * users , const char * executable ) {
  DIR * dir = opendir( DEFAULT_VAR_DIR );
  if (dir != NULL) {
    struct dirent * dp;
    do {
      dp = readdir(dir);
      if (dp != NULL) {
        int pid;
        if (util_sscanf_int( dp->d_name , &pid )) {
          char * full_path = util_alloc_filename( DEFAULT_VAR_DIR , dp->d_name , NULL );
          bool add_user    = false;
          int  uid;
          
          {
            FILE * stream    = util_fopen( full_path , "r");
            char this_executable[512];

            if (fscanf( stream , "%s %d" , this_executable , &uid) == 2) {
              if (executable != NULL) {
                if (util_string_equal( this_executable , executable )) 
                  add_user   = true;
              } else
                add_user = true;
            }
            fclose( stream );
          }
          
          
          /* Remove the pid files of dead processes. */
          if (!util_proc_alive( pid )) {
            unlink( full_path );
            add_user = false;
          }
          
          
          if (add_user) {
            struct passwd *pwd;
            pwd = getpwuid( uid );
            if (pwd != NULL)
              set_add_key( users , pwd->pw_name );
          } 


          free( full_path );
        }
      }
    } while (dp != NULL );
    closedir( dir );
  } 
}
