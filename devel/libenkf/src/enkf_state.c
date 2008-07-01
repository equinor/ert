#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <list.h>
#include <hash.h>
#include <fortio.h>
#include <util.h>
#include <ecl_kw.h>
#include <ecl_block.h>
#include <ecl_fstate.h>
#include <list_node.h>
#include <enkf_node.h>
#include <enkf_state.h>
#include <enkf_types.h>
#include <ecl_static_kw.h>
#include <field.h>
#include <field_config.h>
#include <ecl_util.h>
#include <thread_pool.h>
#include <path_fmt.h>
#include <gen_kw.h>
#include <ecl_sum.h>
#include <well.h>
#include <summary.h>
#include <multz.h>
#include <multflt.h>
#include <equil.h>
#include <well_config.h>
#include <void_arg.h>
#include <pgbox_config.h>
#include <pgbox.h>
#include <restart_kw_list.h>
#include <enkf_fs.h>
#include <meas_vector.h>
#include <enkf_obs.h>
#include <obs_node.h>
#include <basic_driver.h>
#include <enkf_config.h>
#include <node_ctype.h>
#include <job_queue.h>
#include <sched_file.h>
#include <basic_queue_driver.h>
#include <pthread.h>
#include <ext_joblist.h>




/** THE MOST IMPORTANT ENKF OBJECT */

struct enkf_state_struct {
  restart_kw_list_type  * restart_kw_list;   /* This is an ordered list of the keywords in the restart file - to be
                                                able to regenerate restart files with keywords in the right order.*/
  list_type    	   	* node_list;        
  hash_type    	   	* node_hash;
  hash_type             * data_kw;           /* This a list of key - value pairs which are used in a search-replace
                                                operation on the ECLIPSE data file. Will at least contain the key "INIT"
                                                - which will describe initialization of ECLIPSE (EQUIL or RESTART).*/


  meas_vector_type      * meas_vector;       /* The meas_vector will accept measurements performed on this state.*/
  enkf_config_type 	* config;            /* The main config object, which contains all the node config objects. */
  char             	* eclbase;           /* What is the basename (in ECLIPSE) for this state. */
  char                  * run_path;          /* Where should the ECLIPSE simulations be run. */ 
  int                     my_iens;           /* The ensemble number of this instance (not generally zero offset - i.e.
                                                should *NOT* be used as an index .*/
  state_enum              analysis_state;       
  int                     report_step;       /* The report_step and analysis_state = (analyzed || forecast) say
                                                what is the current state with respect to enkf of the in-memory
                                                representation of this enkf_state instance.*/
  char                  * ecl_store_path;    /* Where should the ECLIPSE results be copied (can be NULL in case ecl_store == 0). */
  ecl_store_enum          ecl_store;         /* What should be stored of ECLIPSE results - see the definition
                                                of ecl_store_enum in enkf_types.h. */  
  enkf_fs_type          * fs;                
  ext_joblist_type      * joblist;                
};




static void enkf_state_add_node_internal(enkf_state_type * , const char * , const enkf_node_type * );


/*****************************************************************/

#define ENKF_STATE_APPLY(node_func)                                      \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , int mask) { \
  list_node_type *list_node;                                             \
  list_node = list_get_head(enkf_state->node_list);                      \
  while (list_node != NULL) {                                            \
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          \
    if (enkf_node_include_type(enkf_node , mask))                        \
      enkf_node_ ## node_func (enkf_node);                               \
    list_node = list_node_get_next(list_node);                           \
  }                                                                      \
}


#define ENKF_STATE_APPLY_IENS(node_func)                                 \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , int mask) { \
  list_node_type *list_node;                                             \
  list_node = list_get_head(enkf_state->node_list);                      \
  while (list_node != NULL) {                                            \
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          \
    if (enkf_node_has_func(enkf_node , initialize_func)) {               \
      if (enkf_node_include_type(enkf_node , mask))                      \
         enkf_node_ ## node_func (enkf_node , enkf_state->my_iens);      \
    }                                                                    \
    list_node = list_node_get_next(list_node);                           \
  }                                                                      \
}


void enkf_state_apply_NEW2(enkf_state_type * enkf_state , int mask , node_function_type function_type) {
  list_node_type *list_node;                                             
  list_node = list_get_head(enkf_state->node_list);                      
  while (list_node != NULL) {                                            
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          
    if (enkf_node_include_type(enkf_node , mask)) {
      switch(function_type) {
      case(initialize_func):
	enkf_node_initialize(enkf_node , enkf_state->my_iens);
	break;
      default:
	util_abort("%s . function not implemented ... \n",__func__);
      }
    }
    list_node = list_node_get_next(list_node);                           
  }                                                                      
}


/**
   This function initializes all parameters, either by loading
   from files, or by sampling internally. They are then written to
   file.
*/

void enkf_state_initialize(enkf_state_type * enkf_state) {
  enkf_state_apply_NEW2(enkf_state , parameter , initialize_func);
  enkf_state_fwrite_as(enkf_state , all_types - ecl_restart - ecl_summary , 0 , analyzed);
}


void enkf_state_apply_NEW(enkf_state_type * enkf_state , int mask , enkf_node_ftype_NEW * node_func , void_arg_type * arg) {
  enkf_node_type * enkf_node;
  bool cont;
  enkf_node = hash_iter_get_first(enkf_state->node_hash , &cont);
  while (cont) {
    if (enkf_node_include_type(enkf_node , mask))                        
      node_func(enkf_node , arg);                               
    enkf_node = hash_iter_get_next(enkf_state->node_hash , &cont);
  }                                                                      
}



void enkf_state_apply(enkf_state_type * enkf_state , enkf_node_ftype1 * node_func , int mask) {
  list_node_type *list_node;                                             
  list_node = list_get_head(enkf_state->node_list);                      
  while (list_node != NULL) {                                            
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          
    if (enkf_node_include_type(enkf_node , mask))                        
      node_func (enkf_node);                               
    list_node = list_node_get_next(list_node);                           
  }                                                                      
}


/*****************************************************************/

#define ENKF_STATE_APPLY2(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , const enkf_state_type *enkf_state2 , int mask) { \
  list_node_type *list_node;                                            \
  list_node_type *list_node2;                                           \
  list_node  = list_get_head(enkf_state->node_list);                    \
  list_node2 = list_get_head(enkf_state2->node_list);                   \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    enkf_node_type *enkf_node2 = list_node_value_ptr(list_node2);       \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , enkf_node2);                 \
    list_node  = list_node_get_next(list_node);                         \
    list_node2 = list_node_get_next(list_node2);                        \
  }                                                                     \
}

/*****************************************************************/

#define ENKF_STATE_APPLY_SCALAR(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , double scalar, int mask) { \
  list_node_type *list_node;                                            \
  list_node  = list_get_head(enkf_state->node_list);                    \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , scalar);                     \
    list_node  = list_node_get_next(list_node);                         \
  }                                                                     \
}

/*****************************************************************/

#define ENKF_STATE_APPLY_PATH(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , const char *path, int mask) { \
  list_node_type *list_node;                                            \
  list_node  = list_get_head(enkf_state->node_list);                    \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , path);                       \
    list_node  = list_node_get_next(list_node);                         \
  }                                                                     \
}

/*****************************************************************/



state_enum enkf_state_get_analysis_state(const enkf_state_type * enkf_state) {
  return enkf_state->analysis_state;
}


void enkf_state_set_state(enkf_state_type * enkf_state , int report_step , state_enum state) {
  enkf_state->analysis_state = state;
  enkf_state->report_step    = report_step;
}


bool enkf_state_fmt_file(const enkf_state_type * enkf_state) {
  return enkf_config_get_fmt_file(enkf_state->config);
}


int enkf_state_fmt_mode(const enkf_state_type * enkf_state) {
  if (enkf_state_fmt_file(enkf_state))
    return ECL_FORMATTED;
  else
    return ECL_BINARY;
}


void enkf_state_set_run_path(enkf_state_type * enkf_state , const char * run_path) {
  enkf_state->run_path = util_realloc_string_copy(enkf_state->run_path , run_path);
}


void enkf_state_set_eclbase(enkf_state_type * enkf_state , const char * eclbase) {
  enkf_state->eclbase = util_realloc_string_copy(enkf_state->eclbase , eclbase);
}


void enkf_state_set_iens(enkf_state_type * enkf_state , int iens) {
  enkf_state->my_iens = iens;
}


int  enkf_state_get_iens(const enkf_state_type * enkf_state) {
  return enkf_state->my_iens;
}


enkf_fs_type * enkf_state_get_fs_ref(const enkf_state_type * state) {
  return state->fs;
}


enkf_state_type * enkf_state_alloc(const enkf_config_type * config , int iens , ecl_store_enum ecl_store , enkf_fs_type * fs , ext_joblist_type * joblist , 
				   const char * run_path , const char * eclbase ,  const char * ecl_store_path , meas_vector_type * meas_vector) {
  enkf_state_type * enkf_state = util_malloc(sizeof *enkf_state , __func__);
  
  enkf_state->config          = (enkf_config_type *) config;
  enkf_state->node_list       = list_alloc();
  enkf_state->node_hash       = hash_alloc();
  enkf_state->restart_kw_list = restart_kw_list_alloc();
  enkf_state_set_iens(enkf_state , iens);
  enkf_state->run_path        = NULL;
  enkf_state->eclbase         = NULL;
  enkf_state->fs              = fs;
  enkf_state->joblist         = joblist;
  enkf_state->meas_vector     = meas_vector;
  enkf_state->data_kw         = hash_alloc();
  enkf_state_set_run_path(enkf_state , run_path);
  enkf_state_set_eclbase(enkf_state , eclbase);
  enkf_state->ecl_store_path  = util_alloc_string_copy(ecl_store_path);
  enkf_state->ecl_store       = ecl_store; 
  enkf_state_set_state(enkf_state , -1 , forecast);
  return enkf_state;
}

/*
  hash_node -> list_node -> enkf_node -> {Actual enkf object: multz_type/equil_type/multflt_type/...}
*/




enkf_state_type * enkf_state_copyc(const enkf_state_type * src) {
  enkf_state_type * new = enkf_state_alloc(src->config , src->my_iens, src->ecl_store , src->fs , src->joblist , src->run_path , src->eclbase ,  src->ecl_store_path , src->meas_vector);
  list_node_type *list_node;                                          
  list_node = list_get_head(src->node_list);                     

  while (list_node != NULL) {                                           
    {
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);         
      enkf_node_type *new_node  = enkf_node_copyc(enkf_node);
      enkf_state_add_node_internal(new , enkf_node_get_key_ref(new_node) , new_node);
      list_node = list_node_get_next(list_node);                          
    }
  }
  
  return new;
}



static bool enkf_state_has_node(const enkf_state_type * enkf_state , const char * node_key) {
  return hash_has_key(enkf_state->node_hash , node_key);
}



static void enkf_state_add_node_internal(enkf_state_type * enkf_state , const char * node_key , const enkf_node_type * node) {
  if (enkf_state_has_node(enkf_state , node_key)) 
    util_abort("%s: node:%s already added  - aborting \n",__func__ , node_key);
  {
    list_node_type *list_node = list_append_list_owned_ref(enkf_state->node_list , node , enkf_node_free__);
    /*
      The hash contains a pointer to a list_node structure, which contains a pointer
      to an enkf_node which contains a pointer to the actual enkf object.
    */
    hash_insert_ref(enkf_state->node_hash , node_key , list_node);
  }
}



void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_key , const enkf_config_node_type * config) {
  enkf_node_type *enkf_node = enkf_node_alloc(node_key , config);
  enkf_state_add_node_internal(enkf_state , node_key , enkf_node);    

  /* All code below here is special code for plurigaussian fields */
  {
    enkf_impl_type impl_type = enkf_config_node_get_impl_type(config);
    if (impl_type == PGBOX) {
      const pgbox_config_type * pgbox_config = enkf_config_node_get_ref(config);
      const char * target_key = pgbox_config_get_target_key(pgbox_config);
      if (enkf_state_has_node(enkf_state , target_key)) {
	enkf_node_type * target_node = enkf_state_get_node(enkf_state , target_key);
	if (enkf_node_get_impl_type(target_node) != FIELD) 
	  util_abort("%s: target node:%s is not of type field - aborting \n",__func__ , target_key);
	
	pgbox_set_target_field(enkf_node_value_ptr(enkf_node) , enkf_node_value_ptr(target_node));
      } else 
	util_abort("%s: target field:%s must be added to the state object *BEFORE* the pgbox object - aborting \n" , __func__ , target_key);
    }
  }
}


/*
void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_name) {
  if (enkf_state_has_node(enkf_state , node_name)) {
    fprintf(stderr,"%s: node:%s already added  - aborting \n",__func__ , node_name);
    abort();
  }

  if (!enkf_ensemble_has_key(enkf_state->config , node_name)) {
    fprintf(stderr,"%s could not find configuration object for:%s - aborting \n",__func__ , node_name);
    abort();
    }
    {
    const enkf_config_node_type *config  = enkf_ensemble_get_config_ref(enkf_state->config  , node_name);
    enkf_state_add_node__1(enkf_state , node_name , config);
  }
  }
*/


static void enkf_state_ecl_store(const enkf_state_type * enkf_state , int report_nr1 , int report_nr2) {
  const bool fmt_file  = enkf_state_fmt_file(enkf_state);
  int first_report;
  if (enkf_state->ecl_store != store_none) {

    util_make_path(enkf_state->ecl_store_path);
    if (enkf_state->ecl_store & store_data) {
      char * data_target = ecl_util_alloc_filename(enkf_state->ecl_store_path , enkf_state->eclbase , ecl_data_file , true , -1);
      char * data_src    = ecl_util_alloc_filename(enkf_state->run_path       , enkf_state->eclbase , ecl_data_file , true , -1);
      
      util_copy_file(data_src , data_target);
      free(data_target);
      free(data_src);
    }


    if (enkf_state->ecl_store & store_summary) {
      first_report       = report_nr1 + 1;
      {
	char ** summary_target = ecl_util_alloc_filelist(enkf_state->ecl_store_path , enkf_state->eclbase , ecl_summary_file         , fmt_file , first_report, report_nr2);
	char ** summary_src    = ecl_util_alloc_filelist(enkf_state->run_path       , enkf_state->eclbase , ecl_summary_file         , fmt_file , first_report, report_nr2);
	char  * header_target  = ecl_util_alloc_filename(enkf_state->ecl_store_path , enkf_state->eclbase , ecl_summary_header_file  , fmt_file , report_nr2);
	int i;
	for (i=0; i  < report_nr2 - first_report + 1; i++) 
	  util_copy_file(summary_src[i] , summary_target[i]);

	if (!util_file_exists(header_target)) {
	  char * header_src = ecl_util_alloc_filename(enkf_state->run_path , enkf_state->eclbase , ecl_summary_header_file  , fmt_file , report_nr2);
	  util_copy_file(header_src , header_target);
	  free(header_src);
	}
	util_free_stringlist(summary_target , report_nr2 - first_report + 1);
	util_free_stringlist(summary_src    , report_nr2 - first_report + 1);
	free(header_target);
      }
    }
  
    if (enkf_state->ecl_store & store_restart) {
      if (report_nr1 == 0)
	first_report = 0;
      else
	first_report = report_nr1 + 1;
      {
	char ** restart_target = ecl_util_alloc_filelist(enkf_state->ecl_store_path , enkf_state->eclbase , ecl_restart_file , fmt_file , first_report, report_nr2);
	char ** restart_src    = ecl_util_alloc_filelist(enkf_state->run_path       , enkf_state->eclbase , ecl_restart_file , fmt_file , first_report, report_nr2);
	int i;
	for (i=0; i  < report_nr2 - first_report + 1; i++) 
	  util_copy_file(restart_src[i] , restart_target[i]);

	util_free_stringlist(restart_target , report_nr2 - first_report + 1);
	util_free_stringlist(restart_src    , report_nr2 - first_report + 1);
      }
    }
  }
}

/**
   This function iterates over all the keywords in the ecl_block which
   is input to function. As it iterates it: 
   
   1. It (re)builds the restart_kw_list() object, to be sure that
      we can output these keywords in the correct order.

   2. If the enkf_state does not have a node with the particular kw,
      it is classified as a static keyword, and added to the
      enkf_state object.

   3. The actual data is loaded by calling the enkf_node_load_ecl()
      function.
*/
   

static void enkf_state_load_ecl_restart_block(enkf_state_type * enkf_state , const ecl_block_type *ecl_block) {
  int report_step = ecl_block_get_report_nr(ecl_block);
  ecl_kw_type * ecl_kw = ecl_block_get_first_kw(ecl_block);
  restart_kw_list_reset(enkf_state->restart_kw_list);
  while (ecl_kw != NULL) {
    char *kw = ecl_kw_alloc_strip_header(ecl_kw);
    ecl_util_escape_kw(kw);

    if (enkf_config_has_key(enkf_state->config , kw)) {
      restart_kw_list_add(enkf_state->restart_kw_list , kw);
      /* It is a dynamic restart kw like PRES or SGAS */
      if (enkf_config_impl_type(enkf_state->config , kw) != FIELD) 
	util_abort("%s: hm - something wrong - can (currently) only load fields from restart files - aborting \n",__func__);
      {
	enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw);
	enkf_node_ensure_memory(enkf_node);
	field_copy_ecl_kw_data(enkf_node_value_ptr(enkf_node) , ecl_kw);
      }
    } else {
      /* It is a static kw like INTEHEAD or SCON */
      if (enkf_config_include_static_kw(enkf_state->config , kw)) {
	restart_kw_list_add(enkf_state->restart_kw_list , kw);
	if (!enkf_state_has_node(enkf_state , kw)) 
	  enkf_state_add_node(enkf_state , kw , NULL); 
	{
	  enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw);
	  enkf_node_load_static_ecl_kw(enkf_node , ecl_kw);
	  /*
	    Static kewyords go straight out ....
	  */
	  enkf_fs_fwrite_node(enkf_state->fs , enkf_node , report_step , enkf_state->my_iens , forecast);
	  enkf_node_free_data(enkf_node);
	}
      } 
    }
    free(kw);
    ecl_kw = ecl_block_get_next_kw(ecl_block);
  }
  enkf_fs_fwrite_restart_kw_list(enkf_state->fs , report_step , enkf_state->my_iens, enkf_state->restart_kw_list);
}




void enkf_state_load_ecl_restart(enkf_state_type * enkf_state ,  bool unified , int report_step) {
  bool at_eof;
  const bool fmt_file  = enkf_state_fmt_file(enkf_state);
  bool endian_swap     = enkf_config_get_endian_swap(enkf_state->config);
  ecl_block_type       * ecl_block;
  char * restart_file  = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_restart_file , fmt_file , report_step);

  fortio_type * fortio = fortio_fopen(restart_file , "r" , endian_swap);
  
  if (unified)
    ecl_block_fseek(report_step , fmt_file , true , fortio);
  
  ecl_block = ecl_block_alloc(report_step , fmt_file , endian_swap);
  ecl_block_fread(ecl_block , fortio , &at_eof);
  fortio_fclose(fortio);
  
  enkf_state_load_ecl_restart_block(enkf_state , ecl_block);
  ecl_block_free(ecl_block);
  free(restart_file);
}






   

static void enkf_state_apply_ecl_load(enkf_state_type * enkf_state, int report_step) {
  const bool fmt_file = enkf_state_fmt_file(enkf_state);
  ecl_sum_type * ecl_sum;
  char * summary_file     = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_summary_file        , fmt_file ,  report_step);
  char * header_file      = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_summary_header_file , fmt_file , -1);
  
  ecl_sum = ecl_sum_fread_alloc(header_file , 1 , (const char **) &summary_file , true , enkf_config_get_endian_swap(enkf_state->config));
  /*
    for (iwell = 0; iwell < Nwells; iwell++) {
    if (! enkf_state_has_node(enkf_state , well_list[iwell])) 
    enkf_state_add_node(enkf_state , well_list[iwell] , enkf_config_get_node_ref(enkf_state->config , well_list[iwell])); 
    {
    enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , well_list[iwell]);
    enkf_node_ecl_load(enkf_node , report_step , NULL , ecl_sum);
    }
    }
  */
  
  {
    list_node_type *list_node;                                             
    list_node = list_get_head(enkf_state->node_list);                      
    while (list_node != NULL) {                                            
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);          
      if (enkf_node_has_func(enkf_node , ecl_load_func)) 
	enkf_node_ecl_load(enkf_node , enkf_state->run_path , enkf_state->eclbase , ecl_sum , report_step);
      
      list_node = list_node_get_next(list_node);                           
    }                                                                      
  }
  
  ecl_sum_free(ecl_sum);
  free(summary_file);
  free(header_file);
}


/**
  This function iterates over the observations, and as such it requires
  quite intimate knowledge of enkf_obs_type structure - not quite
  nice.
*/
void enkf_state_measure( const enkf_state_type * enkf_state , enkf_obs_type * enkf_obs) {
  char **obs_keys   	= hash_alloc_keylist(enkf_obs->obs_hash);
  int iobs;

  for (iobs = 0; iobs < hash_get_size(enkf_obs->obs_hash); iobs++) {
    const char * kw = obs_keys[iobs];
    {
      obs_node_type  * obs_node  = hash_get(enkf_obs->obs_hash , kw);
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , obs_node_get_state_kw(obs_node));

      if (!enkf_node_memory_allocated(enkf_node))
	enkf_fs_fread_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
      
      obs_node_measure(obs_node , enkf_state->report_step , enkf_node , enkf_state_get_meas_vector(enkf_state));
    }
  }
  hash_free_ext_keylist(enkf_obs->obs_hash , obs_keys);
}



/*
  Loading of ECLIPSE results goes like this:

  1. First the restart file is loaded, and the various ecl_kw
     instances are distributed depending on whether they are static
     (i.e INTEHEAD and SCONS, ...) or dynamic (i.e. PRESSURE, SWAT ,
     ...). 

  2. Then the function enkf_state_apply_ecl_load() is called; this
     function iterates over all the nodes in the enkf_state(), and
     calls their respective load_ecl() functions (if they have
     one). 
*/



void enkf_state_ecl_load(enkf_state_type * enkf_state , enkf_obs_type * enkf_obs , bool unified , int report_step1 , int report_step2) {
  enkf_state_ecl_store(enkf_state , report_step1 , report_step2);

  /*
    Loading in the X0000 files containing the initial distribution of
    pressure/saturations/....
  */

  if (report_step1 == 0) {
    enkf_state_load_ecl_restart(enkf_state , unified , report_step1);
    enkf_state_fwrite_as(enkf_state , ecl_restart , report_step1 , analyzed);
  }

  enkf_state_set_state(enkf_state , report_step2 , forecast); 
  enkf_state_load_ecl_restart(enkf_state , unified , report_step2);
  enkf_state_apply_ecl_load(enkf_state , report_step2);

  enkf_state_measure(enkf_state , enkf_obs);
  enkf_state_swapout(enkf_state , ecl_restart + ecl_summary);
}




/*
void * enkf_state_ecl_load__(void * input_arg) {
  void_arg_type * void_arg     =  void_arg_safe_cast(input_arg);
  enkf_state_type * enkf_state =  void_arg_get_ptr(void_arg   , 0);
  enkf_obs_type * enkf_obs     =  void_arg_get_ptr(void_arg   , 1);
  int report_step1             =  void_arg_get_int(void_arg   , 2);
  int report_step2             =  void_arg_get_int(void_arg   , 3);
  bool unified                 =  void_arg_get_bool(void_arg  , 4);  
  
  enkf_state_ecl_load(enkf_state , enkf_obs , unified , report_step1 , report_step2);
  return NULL;
}
*/


void enkf_state_write_restart_file(enkf_state_type * enkf_state) {
  bool endian_swap       = enkf_config_get_endian_swap(enkf_state->config);
  const bool fmt_file    = enkf_state_fmt_file(enkf_state);
  char * restart_file    = ecl_util_alloc_filename(enkf_state->run_path , enkf_state->eclbase , ecl_restart_file , fmt_file , enkf_state->report_step);
  fortio_type * fortio   = fortio_fopen(restart_file , "w" , endian_swap);
  const char * kw;

  kw = restart_kw_list_get_first(enkf_state->restart_kw_list);
  while (kw != NULL) {
    enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw);
    enkf_var_type    var_type  = enkf_node_get_var_type(enkf_node);
    if (!enkf_node_memory_allocated(enkf_node))
      enkf_fs_fread_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);  

    if (var_type == ecl_restart) {
      /* Pressure and saturations */
      if (enkf_node_get_impl_type(enkf_node) == FIELD)
	enkf_node_ecl_write_fortio(enkf_node , fortio , fmt_file , FIELD);
      else 
	util_abort("%s: internal error wrong implementetion type:%d - node:%s aborting \n",__func__ , enkf_node_get_impl_type(enkf_node) , enkf_node_get_key_ref(enkf_node));
      
    } else if (var_type == ecl_static) {
      enkf_node_ecl_write_fortio(enkf_node , fortio , fmt_file , STATIC );
      enkf_node_free_data(enkf_node); /* Just immediately discard the static data. */
    }

    kw = restart_kw_list_get_next(enkf_state->restart_kw_list);
  }
  fortio_fclose(fortio);
}


/**
  This function writes out all the files needed by an ECLIPSE
  simulation, this includes the restart file, and the various INCLUDE
  files corresponding to parameteres estimated by EnKF.

  The writing of restart file is delegated to
  enkf_state_write_restart_file().
*/

void enkf_state_ecl_write(enkf_state_type * enkf_state ,  int mask) {
  int    restart_mask    = 0;
  list_node_type *list_node;                                            

  if (mask & ecl_restart) 
    restart_mask += ecl_restart;
  if (mask & ecl_static)
    restart_mask += ecl_static;
  mask -= restart_mask;

  if (restart_mask > 0 && enkf_state->report_step > 0)
    enkf_state_write_restart_file(enkf_state);
  
  
  util_make_path(enkf_state->run_path);
  list_node = list_get_head(enkf_state->node_list);                     
  while (list_node != NULL) {                                           
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);         
    if (enkf_node_include_type(enkf_node , mask)) {

      if (!enkf_node_memory_allocated(enkf_node))
	enkf_fs_fread_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
      
      if (enkf_node_include_type(enkf_node , parameter + static_parameter))
        enkf_node_ecl_write(enkf_node , enkf_state->run_path);
    }
    list_node = list_node_get_next(list_node);
  }
}


/**
  This function takes a report_step and a analyzed|forecast state as
  input; the enkf_state instance is set accordingly and written to
  disk.
*/
void enkf_state_fwrite_as(enkf_state_type * enkf_state  , int mask , int report_step , state_enum state) {
  enkf_state_set_state(enkf_state , report_step , state);
  enkf_state_fwrite(enkf_state , mask);
}


void enkf_state_fwrite(const enkf_state_type * enkf_state , int mask) {
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = (enkf_node_type *) list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_fs_fwrite_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}


void enkf_state_fread(enkf_state_type * enkf_state , int mask , int report_step , state_enum state) {
  list_node_type *list_node;                                            
  enkf_state_set_state(enkf_state , report_step , state);
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = (enkf_node_type *) list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_fs_fread_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}



void enkf_state_swapout(enkf_state_type * enkf_state , int mask ) {
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask)) {                     
      enkf_fs_fwrite_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
      enkf_node_free_data(enkf_node);
    }
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}


void enkf_state_swapin(enkf_state_type * enkf_state , int mask ) {
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask)) {
      enkf_node_ensure_memory(enkf_node);
      enkf_fs_fread_node(enkf_state->fs , enkf_node , enkf_state->report_step , enkf_state->my_iens , enkf_state->analysis_state);
    }
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}



void enkf_state_free_nodes(enkf_state_type * enkf_state, int mask) {
  list_node_type *list_node;                                            
  list_node = list_get_head(enkf_state->node_list);                     
  while (list_node != NULL) {                                           
    list_node_type * next_node = list_node_get_next(list_node);
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);         
    
    if (enkf_node_include_type(enkf_node , mask))      
      enkf_state_del_node(enkf_state , enkf_node_get_key_ref(enkf_node));
    
    list_node = next_node;
  } 
}


/*
void enkf_state_serialize(enkf_state_type * enkf_state , size_t stride) {
  {
    list_node_type *list_node;                                            
    list_node  = list_get_head(enkf_state->node_list);                    
    size_t offset = 0;
    size_t serial_data_size = 100;
    while (list_node != NULL) {                                           
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
      if (enkf_node_include_type(enkf_node , parameter + ecl_restart + ecl_summary))
	offset += stride * enkf_node_serialize(enkf_node , serial_data_size , enkf_state->serial_data , stride , offset);                       
      list_node  = list_node_get_next(list_node);                         
    }             
  }
}
*/

meas_vector_type * enkf_state_get_meas_vector(const enkf_state_type *state) {
  return state->meas_vector;
}


void enkf_state_free(enkf_state_type *enkf_state) {
  list_free(enkf_state->node_list);
  hash_free(enkf_state->node_hash);
  hash_free(enkf_state->data_kw);
  free(enkf_state->run_path);
  restart_kw_list_free(enkf_state->restart_kw_list);
  free(enkf_state->eclbase);
  if (enkf_state->ecl_store_path != NULL) free(enkf_state->ecl_store_path);
  free(enkf_state);
}



enkf_node_type * enkf_state_get_node(const enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    list_node_type * list_node = hash_get(enkf_state->node_hash , node_key);
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);
    return enkf_node;
  } else {
    util_abort("%s: node:%s not found in state object - aborting \n",__func__ , node_key);
    return NULL; /* Compiler shut up */
  }
}



void enkf_state_del_node(enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    list_node_type * list_node = hash_get(enkf_state->node_hash , node_key);
    
    hash_del(enkf_state->node_hash , node_key);
    list_del_node(enkf_state->node_list , list_node);
    
  } else 
    util_abort("%s: node:%s not found in state object - aborting \n",__func__ , node_key);
}


/**
   The value is string - the hash routine takes a copy of the string,
   which means that the calling unit is free to whatever it wants with
   the string.
*/


void enkf_state_set_data_kw(enkf_state_type * enkf_state , const char * kw , const char * value) {
  void_arg_type * void_arg = void_arg_alloc_buffer(strlen(value) + 1, value);
  hash_insert_hash_owned_ref(enkf_state->data_kw , kw , void_arg , void_arg_free__);
}




/**
   init_step    : The parameters are loaded from this EnKF/report step.
   report_step1 : The simulation should start from this report step.
   report_step2 : The simulation should stop at this report step.

   For a normal EnKF run we well have init_step == report_step1, but
   in the case where we want rerun from the beginning with updated
   parameters, they will be different. If init_step != report_step1,
   it is required that report_step1 == 0; otherwise the dynamic data
   will become completely inconsistent. We just don't allow that!
*/


void enkf_state_init_eclipse(enkf_state_type *enkf_state, const sched_file_type * sched_file , int init_step, int report_step1 , int report_step2) {
  
  if (report_step1 != init_step)
    if (report_step1 > 0)
      util_abort("%s: internal error - when initializing from a different timestep than starting from - the start step must be zero.\n",__func__);

  if (report_step1 > 0) {
    char * data_initialize = util_alloc_sprintf("RESTART\n   \'%s\'  %d  /\n" , enkf_state->eclbase , report_step1);
    enkf_state_set_data_kw(enkf_state , "INIT" , data_initialize);
    free(data_initialize);
  }

  util_make_path(enkf_state->run_path);
  {
    char * data_file = ecl_util_alloc_filename(enkf_state->run_path , enkf_state->eclbase , ecl_data_file , true , -1);
    util_filter_file(enkf_config_get_data_file(enkf_state->config) , NULL , data_file , '<' , '>' , enkf_state->data_kw , false);
    free(data_file);
  }

  {
    char * schedule_file = util_alloc_full_path(enkf_state->run_path , enkf_config_get_schedule_target_file(enkf_state->config));
    sched_file_fprintf(sched_file , report_step2 , -1 , -1 , schedule_file);
    free(schedule_file);
  }


  {
    int load_mask = constant + static_parameter + parameter;
    if (report_step1 > 0) load_mask += ecl_restart; 
    enkf_state_fread(enkf_state , load_mask , init_step , analyzed);
  }
  /* 
     Uncertain about this one ...
  */
  enkf_state_set_state(enkf_state , report_step1 , analyzed); 
  enkf_state_ecl_write(enkf_state , constant + static_parameter + parameter + ecl_restart + ecl_static);
  {
    char * stdin_file = util_alloc_full_path(enkf_state->run_path , "eclipse.stdin" );  /* The name eclipse.stdin must be mathched when the job is dispatched. */
    ecl_util_init_stdin( stdin_file , enkf_state->eclbase );
    free(stdin_file);
  }

  {
    int forward_model_length;
    bool  fmt_file              = enkf_config_get_fmt_file(enkf_state->config);
    const char ** forward_model = enkf_config_get_forward_model(enkf_state->config , &forward_model_length);
    hash_type * context = hash_alloc();
    char * restart_file1   = ecl_util_alloc_filename(NULL , enkf_state->eclbase , ecl_restart_file  	   , fmt_file , report_step1);
    char * restart_file2   = ecl_util_alloc_filename(NULL , enkf_state->eclbase , ecl_restart_file  	   , fmt_file , report_step2);
    char * smspec_file     = ecl_util_alloc_filename(NULL , enkf_state->eclbase , ecl_summary_header_file  , fmt_file , -1);
    char * iens            = util_alloc_sprintf("%d" , enkf_state->my_iens);
    char * ecl_base        = enkf_state->eclbase;
    char * report_step1_s  = util_alloc_sprintf("%d" , report_step1);
    char * report_step2_s  = util_alloc_sprintf("%d" , report_step2);


    hash_insert_hash_owned_ref( context , "REPORT_STEP1"  , void_arg_alloc_ptr( report_step1_s ) , void_arg_free__);
    hash_insert_hash_owned_ref( context , "REPORT_STEP2"  , void_arg_alloc_ptr( report_step2_s ) , void_arg_free__);
    hash_insert_hash_owned_ref( context , "RESTART_FILE1" , void_arg_alloc_ptr( restart_file1 )  , void_arg_free__);
    hash_insert_hash_owned_ref( context , "RESTART_FILE2" , void_arg_alloc_ptr( restart_file2 )  , void_arg_free__);
    hash_insert_hash_owned_ref( context , "SMSPEC_FILE"   , void_arg_alloc_ptr( smspec_file   )  , void_arg_free__);
    hash_insert_hash_owned_ref( context , "ECL_BASE"      , void_arg_alloc_ptr( ecl_base   )     , void_arg_free__);
    hash_insert_hash_owned_ref( context , "IENS"          , void_arg_alloc_ptr( iens   )         , void_arg_free__);
    
    ext_joblist_python_fprintf( enkf_state->joblist , forward_model , forward_model_length , enkf_state->run_path , context);
    
    free(iens);
    free(restart_file1);
    free(restart_file2);
    free(smspec_file);
    free(report_step1_s);
    free(report_step2_s);
  }
}






/**
   xx_run_eclipse() has been split in two functions:

   1: enkf_state_start_eclipse()

   2: enkf_state_complete_eclipse()

   Because the firstis quite CPU intensive (gunzip), and the number of
   concurrent threads should be limitied. For the second there are one
   thread for each ensemble member. This is handeled by the calling scope.
*/



void enkf_state_start_eclipse(enkf_state_type * enkf_state , job_queue_type * job_queue , const sched_file_type * sched_file , int init_step , int report_step1 , int report_step2) {
  const int iens        = enkf_state_get_iens(enkf_state);
  /* 
     Prepare the job and submit it to the queue
  */

  enkf_state_init_eclipse(enkf_state , sched_file , init_step , report_step1 , report_step2);
  job_queue_add_job(job_queue , iens , report_step2);
}


void enkf_state_complete_eclipse(enkf_state_type * enkf_state , job_queue_type * job_queue , enkf_obs_type * enkf_obs , bool unified , int report_step1 , int report_step2 , bool load_results , bool unlink_run_path , bool *job_OK) {
  const int usleep_time = 100000; /* 1/10 of a second */ 
  const int iens        = enkf_state_get_iens(enkf_state);

  *job_OK = true;
  while (true) {
    job_status_type status = job_queue_export_job_status(job_queue , iens);

    if (status == job_queue_complete_OK) {
      if (load_results)
	enkf_state_ecl_load(enkf_state , enkf_obs , unified , report_step1 , report_step2);
      break;
    } else if (status == job_queue_complete_FAIL) {
      fprintf(stderr,"** job:%d failed completely - this will break ... \n",iens);
      *job_OK = false;
      break;
    } else usleep(usleep_time);
  } 
  
  /* In case the job fails, we leave the run_path directory. */
  if (unlink_run_path && (status == job_queue_queue_complete_OK))
    util_unlink_path(enkf_state->run_path);
}


void * enkf_state_complete_eclipse__(void * __void_arg) {
  void_arg_type * void_arg          = void_arg_safe_cast(__void_arg);
  enkf_state_type * enkf_state 	    = void_arg_get_ptr(void_arg    ,  0);
  job_queue_type  * job_queue       = void_arg_get_ptr(void_arg    ,  1);
  enkf_obs_type   * enkf_obs   	    = void_arg_get_ptr(void_arg    ,  2);
  bool              unified    	    = void_arg_get_bool(void_arg   ,  4);
  int               report_step1    = void_arg_get_int(void_arg    ,  6);
  int               report_step2    = void_arg_get_int(void_arg    ,  7);
  bool              load_results    = void_arg_get_bool(void_arg   ,  9);
  bool              unlink_run_path = void_arg_get_bool(void_arg   , 10);
  bool            * job_OK          = void_arg_get_buffer(void_arg , 11);

  enkf_state_complete_eclipse(enkf_state , job_queue , enkf_obs , unified , report_step1 , report_step2 , load_results , unlink_run_path , job_OK);
  return NULL ;
}


void * enkf_state_start_eclipse__(void * __void_arg) {
  void_arg_type * void_arg       = void_arg_safe_cast(__void_arg);
  enkf_state_type * enkf_state 	 = void_arg_get_ptr(void_arg  	, 0);
  job_queue_type  * job_queue    = void_arg_get_ptr(void_arg  	, 1);
  sched_file_type * sched_file   = void_arg_get_ptr(void_arg    , 3);
  int               init_step    = void_arg_get_int(void_arg   , 5);
  int               report_step1 = void_arg_get_int(void_arg  	, 6);
  int               report_step2 = void_arg_get_int(void_arg  	, 7);


  enkf_state_start_eclipse(enkf_state , job_queue , sched_file , init_step , report_step1 , report_step2);
  return NULL ; 
}


int enkf_state_get_report_step(const enkf_state_type * enkf_state) { 
  return enkf_state->report_step;
}



/*****************************************************************/

/**
  Observe that target_serial_size and _serial_size count the number of
  double elements, *NOT* the number of bytes, hence it is essential to
  have a margin to avoid overflow of the size_t datatype (on 32 bit machines).
*/

static double * enkf_ensemble_alloc_serial_data(int ens_size , size_t target_serial_size , size_t * _serial_size) {
  size_t   serial_size = target_serial_size;
  double * serial_data;

#ifdef i386
  /* 
     33570816 = 2^25 is the maximum number of doubles we will
     allocate, this corresponds to 2^28 bytes - which it seems
     we can adress quite safely ...
  */
  serial_size = util_int_min(serial_size , 33570816 ); 
#endif
  
  do {
    serial_data = malloc(serial_size * sizeof * serial_data);
    if (serial_data == NULL) 
      serial_size /= 2;

  } while (serial_data == NULL);

  
  /*
    Ensure that the allocated memory is an integer times ens_size.
  */
  {
    int serial_size0 = serial_size;
    {
      div_t tmp   = div(serial_size , ens_size);
      serial_size = ens_size * tmp.quot;
    }
    if (serial_size != serial_size0) {
      /* Can not use realloc() here because the temporary memory requirements might be prohibitive. */
      free(serial_data);
      serial_data = util_malloc(serial_size * sizeof * serial_data , __func__);
    }
  }
  *_serial_size = serial_size;
  return serial_data;
}




void enkf_ensembleemble_mulX(double * serial_state , int serial_x_stride , int serial_y_stride , int serial_x_size , int serial_y_size , const double * X , int X_x_stride , int X_y_stride) {
  double * line = malloc(serial_x_size * sizeof * line);
  int ix,iy;
  
  for (iy=0; iy < serial_y_size; iy++) {
    if (serial_x_stride == 1) 
      memcpy(line , &serial_state[iy * serial_y_stride] , serial_x_size * sizeof * line);
    else
      for (ix = 0; ix < serial_x_size; ix++)
	line[ix] = serial_state[iy * serial_y_stride + ix * serial_x_stride];

    for (ix = 0; ix < serial_x_size; ix++) {
      int k;
      double dot_product = 0;
      for (k = 0; k < serial_x_size; k++)
	dot_product += line[k] * X[ix * X_x_stride + k*X_y_stride];
      serial_state[ix * serial_x_stride + iy * serial_y_stride] = dot_product;
    }

  }
  
  free(line);
}



void * enkf_ensemble_serialize__(void * _void_arg) {
  void_arg_type * void_arg     = void_arg_safe_cast( _void_arg );
  int update_mask;
  int iens , iens1 , iens2 , serial_stride;
  size_t serial_size;
  double *serial_data;
  size_t * member_serial_size;
  bool   * member_complete;
  list_node_type ** start_node; 
  list_node_type ** next_node;  
  
  iens1       	     = void_arg_get_int(void_arg , 0 );
  iens2       	     = void_arg_get_int(void_arg , 1 );
  serial_size 	     = void_arg_get_size_t(void_arg , 2);
  serial_stride      = void_arg_get_int(void_arg , 3);
  serial_data        = void_arg_get_ptr(void_arg , 4);
  start_node 	     = void_arg_get_ptr(void_arg , 5);
  next_node  	     = void_arg_get_ptr(void_arg , 6);
  member_serial_size = void_arg_get_ptr(void_arg , 7);
  member_complete    = void_arg_get_ptr(void_arg , 8);
  update_mask        = void_arg_get_int(void_arg , 9);
  for (iens = iens1; iens < iens2; iens++) {
    list_node_type  * list_node  = start_node[iens];
    bool node_complete           = true;  
    size_t   serial_offset       = iens;
    
    while (node_complete) {                                           
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
      if (enkf_node_include_type(enkf_node , update_mask)) {                       
	int elements_added        = enkf_node_serialize(enkf_node , serial_size , serial_data , serial_stride , serial_offset , &node_complete);
	serial_offset            += serial_stride * elements_added;  
	member_serial_size[iens] += elements_added;
      }
      
      if (node_complete) {
	list_node  = list_node_get_next(list_node);                         
	if (list_node == NULL) {
	  if (node_complete) member_complete[iens] = true;
	  break;
	}
      }
    }
      /* Restart on this node */
    next_node[iens] = list_node;
  }

  return NULL;
}



void enkf_ensemble_update(enkf_state_type ** enkf_ensemble , int ens_size , size_t target_serial_size , const double * X) {
  const int threads = 1;
  int update_mask = ecl_summary + ecl_restart + parameter;
  thread_pool_type * tp = thread_pool_alloc(0 /* threads */);
  void_arg_type ** void_arg    = malloc(threads * sizeof * void_arg);
  int *     iens1              = malloc(threads * sizeof * iens1);
  int *     iens2              = malloc(threads * sizeof * iens2);
  bool *    member_complete    = malloc(ens_size * sizeof * member_complete);
  size_t  * member_serial_size = malloc(ens_size * sizeof * member_serial_size);
  size_t    serial_size;
  double *  serial_data   = enkf_ensemble_alloc_serial_data(ens_size , target_serial_size , &serial_size);
  int       serial_stride = ens_size;
  int       iens , ithread;
  

  bool      state_complete = false;
  list_node_type  ** start_node = malloc(ens_size * sizeof * start_node);
  list_node_type  ** next_node  = malloc(ens_size * sizeof * next_node);
  
  
  for (iens = 0; iens < ens_size; iens++) {
    enkf_state_type * enkf_state = enkf_ensemble[iens];
    start_node[iens] = list_get_head(enkf_state->node_list);                    
    enkf_state_apply(enkf_ensemble[iens] , enkf_node_clear_serial_state , update_mask);
    member_complete[iens] = false;
  }
  
  {
    int thread_block_size = ens_size / threads;
    for (ithread = 0; ithread < threads; ithread++) {
      iens1[ithread] = ithread * thread_block_size;
      iens2[ithread] = iens1[ithread] + thread_block_size;
      
      void_arg[ithread] = void_arg_alloc10(int_value 	 ,     /* 0 */
					   int_value 	 ,     /* 1 */
					   size_t_value  ,     /* 2 */
					   int_value     ,     /* 3 */
					   void_pointer  ,     /* 4 */
					   void_pointer  ,     /* 5 */
					   void_pointer  ,     /* 6 */
					   void_pointer  ,     /* 7 */
					   void_pointer  ,     /* 8 */ 
					   int_value       );  /* 9 */
    }
    iens2[threads-1] = ens_size;
  }


  while (!state_complete) {
    for (iens = 0; iens < ens_size; iens++) 
      member_serial_size[iens] = 0;
    
    for (ithread =  0; ithread < threads; ithread++) {
      void_arg_pack_int(void_arg[ithread]     , 0 , iens1[ithread]);
      void_arg_pack_int(void_arg[ithread]     , 1 , iens2[ithread]);
      void_arg_pack_size_t(void_arg[ithread]  , 2 , serial_size);
      void_arg_pack_int(void_arg[ithread]     , 3 , serial_stride);
      void_arg_pack_ptr(void_arg[ithread]     , 4 , serial_data);
      void_arg_pack_ptr(void_arg[ithread]     , 5 , start_node);
      void_arg_pack_ptr(void_arg[ithread]     , 6 , next_node);
      void_arg_pack_ptr(void_arg[ithread]     , 7 , member_serial_size);
      void_arg_pack_ptr(void_arg[ithread]     , 8 , member_complete);
      void_arg_pack_int(void_arg[ithread]     , 9 , update_mask);
    }
    
    for (ithread =  0; ithread < threads; ithread++) 
      thread_pool_add_job(tp , &enkf_ensemble_serialize__ , void_arg[ithread]);
    thread_pool_join(tp);

    /* Serialize section */
/*     for (iens = 0; iens < ens_size; iens++) { */
/*       list_node_type  * list_node  = start_node[iens]; */
/*       bool node_complete           = true;   */
/*       size_t   serial_offset       = iens; */
      
/*       while (node_complete) {                                            */
/* 	enkf_node_type *enkf_node = list_node_value_ptr(list_node);         */
/* 	if (enkf_node_include_type(enkf_node , update_mask)) {                        */
/* 	  int elements_added = enkf_node_serialize(enkf_node , serial_size , serial_data , serial_stride , serial_offset , &node_complete); */
/* 	  serial_offset            += serial_stride * elements_added;   */
/* 	  member_serial_size[iens] += elements_added; */
/* 	} */
	
/* 	if (node_complete) { */
/* 	  list_node  = list_node_get_next(list_node);                          */
/* 	  if (list_node == NULL) { */
/* 	    if (node_complete) member_complete[iens] = true; */
/* 	    break; */
/* 	  } */
/* 	} */
/*       } */
/*       /\* Restart on this node *\/ */
/*       next_node[iens] = list_node; */
/*     } */

    for (iens=1; iens < ens_size; iens++) {
      if (member_complete[iens]    != member_complete[iens-1])    util_abort("%s: member_complete difference    - INTERNAL ERROR - aborting \n",__func__); 
      if (member_serial_size[iens] != member_serial_size[iens-1]) util_abort("%s: member_serial_size difference - INTERNAL ERROR - aborting \n",__func__); 
    }
    state_complete = member_complete[0];
    
    

    /* Update section */
    enkf_ensembleemble_mulX(serial_data , 1 , ens_size , ens_size , member_serial_size[0] , X , ens_size , 1);


    /* deserialize section */
    for (iens = 0; iens < ens_size; iens++) {
      list_node_type  * list_node  = start_node[iens];
      
      while (1) {
	enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
	if (enkf_node_include_type(enkf_node , update_mask)) 
	  enkf_node_deserialize(enkf_node , serial_data , serial_stride);
	
	if (list_node == next_node[iens])
	  break;
	
	list_node  = list_node_get_next(list_node);                         
	if (list_node == NULL)
	  break;
      }
    }
    
    for (iens = 0; iens < ens_size; iens++) 
      start_node[iens] = next_node[iens];
  }
  for (ithread = 0; ithread < threads; ithread++) 
    void_arg_free(void_arg[ithread]);
  thread_pool_free(tp);

  free(void_arg);
  free(member_complete);
  free(member_serial_size);
  free(iens1);
  free(iens2);


  free(start_node);
  free(serial_data);
  free(next_node);
}




/*****************************************************************/
/* Generatad functions - iterating through all members.          */
/*****************************************************************/


/*ENKF_STATE_APPLY_PATH(fread);*/
ENKF_STATE_APPLY(clear);
ENKF_STATE_APPLY(clear_serial_state);
ENKF_STATE_APPLY_SCALAR(scale);
ENKF_STATE_APPLY2(imul);
ENKF_STATE_APPLY2(iadd);
ENKF_STATE_APPLY2(iaddsqr);
