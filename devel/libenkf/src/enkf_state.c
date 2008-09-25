#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
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
#include <stringlist.h>

#define ENKF_STATE_TYPE_ID 78132


/**
   This struct is a pure utility structure used to pack the various
   bits and pieces of information needed to start, monitor, and load
   back results from the forward model simulations. 

   Typcially the values in this struct are set from the enkf_main object
   before a forward_step starts.
*/

typedef struct run_info_struct {
  bool              __ready;         	   /* An attempt to check the internal state - not active yet. */
  bool              active;                /* Is this state object active at all - used for instance in ensemble experiments ere only some of the members are integrated. */
  int               init_step;       	   /* The report step we initialize from - will often be equal to step1, but can be different. */
  state_enum        init_state;      	   /* Whether we should init from a forecast or an analyzed state. */
  int               step1;           	   /* The forward model is integrated: step1 -> step2 */
  int               step2;  	     
  bool              load_results;    	   /* Whether the results should be loaded when the forward model is complete. */
  bool              unlink_run_path; 	   /* Whether the run_path should be unlinked when the forward model is through.  [Partly in conflict with member_config->keep_runpath]*/
  const stringlist_type * forward_model;   /* The current forward model - as a list of ext_joblist identifiers (i.e. strings) */
  char             *run_path;              /* The currently used runpath - is realloced / freed for every step. */
  bool              can_sim;               /* If false - this member can not start simulations. */
  
  /******************************************************************/
  /* Return value - set in the called routine!!  */
  bool              complete_OK;     /* Did the forward model complete OK? */

} run_info_type;



/**
   This struct contains various objects which the enkf_state needs
   during operation, which the enkf_state_object *DOES NOT* own. The
   struct only contains pointers to objects owned by (typically) the
   enkf_main object. 

   If the enkf_state object writes to any of the objects in this
   struct that can be considered a serious *BUG*.
*/

typedef struct shared_info_struct {
  enkf_fs_type          * fs;                /* The filesystem object - used to load and store nodes. */
  ext_joblist_type      * joblist;           /* The list of external jobs which are installed - and *how* they should be run (with Python code) */
  sched_file_type       * sched_file;        /* The schedule file. */   
  job_queue_type        * job_queue;         /* The queue handling external jobs. (i.e. LSF / rsh / local / ... )*/ 
  bool                    unified;           /* Use unified ECLIPSE files (NO - do not use that) */
  enkf_obs_type         * obs;               /* The observation struct - used when measuring on the newly loaded state. */
  path_fmt_type         * run_path_fmt;      /* The format specifier for the runpath - when used it is called with integer arguments: iens, report1 , report2 */
} shared_info_type;




/**
   This struct contains information which is private to this
   member. It is initialized at object boot time, and (typically) not
   changed during the simulation. [In principle it could change during
   the simulation, but the current API does not support that.]
*/ 

typedef struct member_config_struct {
  bool             unified;           /* Whether unified ECLIPSE files should be used - NO. */
  char 		 * eclbase;           /* The ECLBASE string used for simulations of this member. */
  int  		   iens;              /* The ensemble member number of this member. */
  bool             keep_runpath;      /* Should the run-path directory be left around (for this member) [Partly in conflict with run_info->unlink_run_path] */
} member_config_type;





/*****************************************************************/
/** THE MOST IMPORTANT ENKF OBJECT */

struct enkf_state_struct {
  int                     __id;              /* Funny integer used for run_time type checking. */
  restart_kw_list_type  * restart_kw_list;   /* This is an ordered list of the keywords in the restart file - to be
                                                able to regenerate restart files with keywords in the right order.*/
  hash_type    	   	* node_hash;
  hash_type             * data_kw;           /* This a list of key - value pairs which are used in a search-replace
                                                operation on the ECLIPSE data file. Will at least contain the key "INIT"
                                                - which will describe initialization of ECLIPSE (EQUIL or RESTART).*/

  meas_vector_type      * meas_vector;       /* The meas_vector will accept measurements performed on this state.*/
  enkf_config_type 	* config;            /* The main config object, which contains all the node config objects. */
  
  run_info_type         * run_info;          /* Various pieces of information needed by the enkf_state object when running the forward model. Updated for each report step.*/
  shared_info_type      * shared_info;       /* Pointers to shared objects which is needed by the enkf_state object (read only). */
  member_config_type    * my_config;         /* Private config information for this member; not updated during a simulation. */
};

/*****************************************************************/


/**
   Currently no locking is implemented - the lock_mode parameter is not used. 
*/

static void run_info_set_run_path(run_info_type * run_info , lock_mode_type lock_mode , int iens , path_fmt_type * run_path_fmt) {
  util_safe_free(run_info->run_path);
  run_info->run_path = path_fmt_alloc_path(run_path_fmt , true , iens , run_info->step1 , run_info->step2);
}



/**
   This function sets the run_info parameters. This is typically called
   (via an enkf_state__ routine) by the external scope handling the forward model.

   When this initialization code has been run we are certain that the
   enkf_state object has all the information it needs to "run itself"
   forward.
*/


static void run_info_set(run_info_type * run_info ,bool active, int init_step , state_enum init_state , int step1 , int step2 , bool load_results , bool unlink_run_path , const stringlist_type * forward_model, int iens , path_fmt_type * run_path_fmt) {
  run_info->active          = active;
  run_info->init_step  	    = init_step;
  run_info->init_state 	    = init_state;
  run_info->step1      	    = step1;
  run_info->step2      	    = step2;
  run_info->load_results    = load_results;
  run_info->unlink_run_path = unlink_run_path;
  run_info->forward_model   = forward_model;
  run_info->complete_OK     = false;
  run_info->__ready         = true;
  run_info->can_sim         = true;
  run_info_set_run_path(run_info , lock_none , iens , run_path_fmt);
}


static run_info_type * run_info_alloc() {
  run_info_type * run_info = util_malloc(sizeof * run_info , __func__);
  run_info->run_path = NULL;
  return run_info;
}


static void run_info_free(run_info_type * run_info) {
  util_safe_free(run_info->run_path);
  free(run_info);
}


static void run_info_complete_run(run_info_type * run_info) {
  if (run_info->complete_OK)
    run_info->run_path = util_safe_free(run_info->run_path);
}


static char * __config_alloc_simlock(const char * lock_path , const char * run_path) {
  char * lockfile;
  {
    char * tmp_path;
    if (util_is_abs_path(run_path))
      tmp_path = util_alloc_string_copy( &run_path[1]);
    else
      tmp_path = util_alloc_string_copy( run_path );
    util_string_tr(tmp_path , UTIL_PATH_SEP_CHAR , '.');
    lockfile = util_alloc_sprintf("%s%clock-%s" , lock_path , UTIL_PATH_SEP_CHAR , tmp_path);
    free(tmp_path);
  }
  return lockfile;
}




/*****************************************************************/

static shared_info_type * shared_info_alloc(enkf_fs_type * enkf_fs , ext_joblist_type * joblist , sched_file_type * sched_file, enkf_obs_type * obs, job_queue_type * job_queue, path_fmt_type * run_path_fmt) {
  shared_info_type * shared_info = util_malloc(sizeof * shared_info , __func__);

  shared_info->fs           = enkf_fs;
  shared_info->joblist      = joblist;
  shared_info->sched_file   = sched_file;
  shared_info->obs          = obs;
  shared_info->job_queue    = job_queue;
  shared_info->run_path_fmt = run_path_fmt;

  return shared_info;
}


static void shared_info_free(shared_info_type * shared_info) {
  /** 
      Adding something here is a BUG - this object does 
      not own anything.
  */
  free( shared_info );
}

                                         

/******************************************************************/
/** Implementation of the member_config struct. All of this implementation
    is private - however some of it is exported through the enkf_state object,
    and it should be perfectly safe to export more of it.
*/


static void member_config_set_eclbase(member_config_type * member_config , const char * eclbase) {
  member_config->eclbase = util_realloc_string_copy(member_config->eclbase , eclbase);
}



static void member_config_free(member_config_type * member_config) {
  util_safe_free(member_config->eclbase);
  free(member_config);
}



static member_config_type * member_config_alloc(int iens , lock_mode_type lock_mode , const char * lock_path , path_fmt_type * run_path_fmt , path_fmt_type * eclbase_fmt , bool keep_runpath) {
  member_config_type * member_config = util_malloc(sizeof * member_config , __func__);
  
  member_config->unified        = false;
  member_config->iens           = iens; /* Can only be changed in the allocater. */
  member_config->eclbase  	= NULL;
  member_config->keep_runpath   = keep_runpath;
  {
    char * eclbase = path_fmt_alloc_path(eclbase_fmt, false , member_config->iens);
    member_config_set_eclbase(member_config  , eclbase);
    free(eclbase);
  }

  return member_config;
}

/*****************************************************************/
/** Helper classes complete - starting on th enkf_state proper object. */
/*****************************************************************/


static void enkf_state_add_node_internal(enkf_state_type * , const char * , const enkf_node_type * );


void enkf_state_apply_NEW2(enkf_state_type * enkf_state , int mask , node_function_type function_type, void_arg_type * arg) {
  hash_lock( enkf_state->node_hash );
  {
    char ** key_list    = hash_alloc_keylist(enkf_state->node_hash);
    int ikey;
    member_config_type * my_config = enkf_state->my_config;
    
    for (ikey = 0; ikey < hash_get_size( enkf_state->node_hash ); ikey++) {
      enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
      if (enkf_node_include_type(enkf_node , mask)) {                       
	switch(function_type) {
	case(initialize_func):
	  enkf_node_initialize(enkf_node , my_config->iens);
	  break;
	case(clear_serial_state_func):
	  enkf_node_clear_serial_state(enkf_node);
	  break;
	default:
	  util_abort("%s . function not implemented ... \n",__func__);
	}
      }
    }                                                                      
  }
  hash_unlock( enkf_state->node_hash );
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


/**
   The run_info->run_path variable is in general NULL. It is given a
   valid value before a simulation starts, holds on to that value
   through the simulation, and is then freed (and set to NULL) when
   the simulation ends.

   The *ONLY* point when an external call to this function should give
   anything is when the forward model has failed, then
   run_info_complete_run() has left the run_path intact.
*/

const char * enkf_state_get_run_path(const enkf_state_type * enkf_state) { 
  return enkf_state->run_info->run_path; 
}




void enkf_state_set_eclbase(enkf_state_type * enkf_state , const char * eclbase) {
  member_config_set_eclbase(enkf_state->my_config , eclbase);
}


/*
  void enkf_state_set_iens(enkf_state_type * enkf_state , int iens) {
  enkf_state->my_iens = iens;
  }
*/

int  enkf_state_get_iens(const enkf_state_type * enkf_state) {
  return enkf_state->my_config->iens;
}


enkf_fs_type * enkf_state_get_fs_ref(const enkf_state_type * state) {
  return state->shared_info->fs;
}


static enkf_state_type * enkf_state_safe_cast(void * __enkf_state) {
  enkf_state_type * state = (enkf_state_type *) __enkf_state;
  if (state->__id != ENKF_STATE_TYPE_ID)
    util_abort("%s: internal error - runtime cast failed - aborting \n",__func__);
  return state;
}



enkf_state_type * enkf_state_alloc(const enkf_config_type * config , int iens , lock_mode_type lock_mode , const char * lock_path , bool keep_runpath,
				   enkf_fs_type * fs , ext_joblist_type * joblist , job_queue_type * job_queue , sched_file_type * sched_file , 
				   path_fmt_type * run_path_fmt , path_fmt_type * eclbase_fmt , meas_vector_type * meas_vector , enkf_obs_type * obs) {
  enkf_state_type * enkf_state = util_malloc(sizeof *enkf_state , __func__);
  enkf_state->__id            = ENKF_STATE_TYPE_ID;
  
  enkf_state->config          = (enkf_config_type *) config;
  enkf_state->node_hash       = hash_alloc();
  enkf_state->restart_kw_list = restart_kw_list_alloc();
  enkf_state->meas_vector     = meas_vector;
  enkf_state->data_kw         = hash_alloc();
  {
    char * iens_string = util_alloc_sprintf("%d" , iens);
    enkf_state_set_data_kw(enkf_state , "IENS" , iens_string);
    free(iens_string);
  }
  enkf_state->my_config   = member_config_alloc( iens , lock_mode , lock_path , run_path_fmt , eclbase_fmt , keep_runpath);
  enkf_state->shared_info = shared_info_alloc(fs , joblist , sched_file , obs , job_queue , run_path_fmt);
  enkf_state->run_info    = run_info_alloc();
  return enkf_state;
}



enkf_state_type * enkf_state_copyc(const enkf_state_type * src) {
  util_abort("%s: not implemented \n",__func__);
  return NULL;
}



static bool enkf_state_has_node(const enkf_state_type * enkf_state , const char * node_key) {
  bool has_node = hash_has_key(enkf_state->node_hash , node_key);
  return has_node;
}



/**
   The enkf_state inserts a reference to the node object. The
   enkf_state object takes ownership of the node object, i.e. it will
   free it when the game is over.
*/


static void enkf_state_add_node_internal(enkf_state_type * enkf_state , const char * node_key , const enkf_node_type * node) {
  if (enkf_state_has_node(enkf_state , node_key)) 
    util_abort("%s: node:%s already added  - aborting \n",__func__ , node_key);
  hash_insert_hash_owned_ref(enkf_state->node_hash , node_key , node, enkf_node_free__);
}



void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_key , const enkf_config_node_type * config) {
  enkf_node_type *enkf_node = enkf_node_alloc(config);
  enkf_state_add_node_internal(enkf_state , node_key , enkf_node);    
}



/*
static void enkf_state_ecl_store(const enkf_state_type * enkf_state , int report_nr1 , int report_nr2) {
  const bool fmt_file  = enkf_state_fmt_file(enkf_state);
  const member_config_type * my_config = enkf_state->my_config;
  int first_report;

  if (my_config->ecl_store != store_none) {
    util_make_path(my_config->ecl_store_path);
    if (my_config->ecl_store & store_data) {
      char * data_target = ecl_util_alloc_filename(my_config->ecl_store_path , my_config->eclbase , ecl_data_file , true , -1);
      char * data_src    = ecl_util_alloc_filename(my_config->run_path       , my_config->eclbase , ecl_data_file , true , -1);
      
      util_copy_file(data_src , data_target);
      free(data_target);
      free(data_src);
    }


    if (my_config->ecl_store & store_summary) {
      first_report       = report_nr1 + 1;
      {
	char ** summary_target = ecl_util_alloc_filelist(my_config->ecl_store_path , my_config->eclbase , ecl_summary_file         , fmt_file , first_report, report_nr2);
	char ** summary_src    = ecl_util_alloc_filelist(my_config->run_path       , my_config->eclbase , ecl_summary_file         , fmt_file , first_report, report_nr2);
	char  * header_target  = ecl_util_alloc_filename(my_config->ecl_store_path , my_config->eclbase , ecl_summary_header_file  , fmt_file , report_nr2);
	int i;
	for (i=0; i  < report_nr2 - first_report + 1; i++) 
	  util_copy_file(summary_src[i] , summary_target[i]);

	if (!util_file_exists(header_target)) {
	  char * header_src = ecl_util_alloc_filename(my_config->run_path , my_config->eclbase , ecl_summary_header_file  , fmt_file , report_nr2);
	  util_copy_file(header_src , header_target);
	  free(header_src);
	}
	util_free_stringlist(summary_target , report_nr2 - first_report + 1);
	util_free_stringlist(summary_src    , report_nr2 - first_report + 1);
	free(header_target);
      }
    }
  
    if (my_config->ecl_store & store_restart) {
      if (report_nr1 == 0)
	first_report = 0;
      else
	first_report = report_nr1 + 1;
      {
	char ** restart_target = ecl_util_alloc_filelist(my_config->ecl_store_path , my_config->eclbase , ecl_restart_file , fmt_file , first_report, report_nr2);
	char ** restart_src    = ecl_util_alloc_filelist(my_config->run_path       , my_config->eclbase , ecl_restart_file , fmt_file , first_report, report_nr2);
	int i;
	for (i=0; i  < report_nr2 - first_report + 1; i++) 
	  util_copy_file(restart_src[i] , restart_target[i]);

	util_free_stringlist(restart_target , report_nr2 - first_report + 1);
	util_free_stringlist(restart_src    , report_nr2 - first_report + 1);
      }
    }
  }
}
*/





/**
  This function iterates over the observations, and as such it requires
  quite intimate knowledge of enkf_obs_type structure - not quite
  nice.

  Observe that this _must_ come after writing to file, because the
  nodes which are measured on are unconditionally loaded from file.
*/
void enkf_state_measure( const enkf_state_type * enkf_state , enkf_obs_type * enkf_obs) {
  const member_config_type * my_config = enkf_state->my_config;
  const run_info_type      * run_info  = enkf_state->run_info;
  char **obs_keys = hash_alloc_keylist(enkf_obs->obs_hash);
  int iobs;

  for (iobs = 0; iobs < hash_get_size(enkf_obs->obs_hash); iobs++) {
    const char * kw = obs_keys[iobs];
    {
      obs_node_type  * obs_node  = hash_get(enkf_obs->obs_hash , kw);
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , obs_node_get_state_kw(obs_node));
      enkf_fs_fread_node(enkf_state->shared_info->fs , enkf_node , run_info->step2 , my_config->iens , forecast); /* Hardcoded to measure on the forecast */
      obs_node_measure(obs_node , run_info->step2 , enkf_node , enkf_state_get_meas_vector(enkf_state));
    }
  }
  util_free_stringlist( obs_keys , hash_get_size( enkf_obs->obs_hash ));
}




/**
   This function loads results from the forward run (i.e. ECLIPSE
   ...). This process is divided in the following steps:

   1. Load an ecl_block instance from the restart file, and a ecl_sum
      instance from the summary file.

   2. Iterate through the ecl_block instance and:

      * Build a restart_kw_list instance.
      * Store all static keywords right away.

   3. Iterating through all the nodes, and call their _ecl_load
      function (through enkf_node_ecl_load).
*/


static void enkf_state_ecl_load2(enkf_state_type * enkf_state ,  bool unified , int mask , int report_step) {
  member_config_type * my_config   = enkf_state->my_config;
  shared_info_type   * shared_info = enkf_state->shared_info;
  run_info_type      * run_info    = enkf_state->run_info;
  const bool fmt_file  = enkf_state_fmt_file(enkf_state);
  bool endian_swap     = enkf_config_get_endian_swap(enkf_state->config);
  ecl_block_type * restart_block = NULL;
  ecl_sum_type   * summary       = NULL;
  
  
  /**
     Loading the restart block.
  */
  if (mask & ecl_restart) {
    char * restart_file  = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ecl_restart_file , fmt_file , report_step);
    fortio_type * fortio = fortio_fopen(restart_file , "r" , endian_swap);
    
    if (unified)
      ecl_block_fseek(report_step , fmt_file , true , fortio);
    
    restart_block = ecl_block_alloc(report_step , fmt_file , endian_swap);
    ecl_block_fread(restart_block , fortio , NULL);
    fortio_fclose(fortio);
    free(restart_file);
  }
  
  
  /**  
     Loading the summary information.
  */
  if (mask & ecl_summary) {
    char * summary_file     = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ecl_summary_file        , fmt_file ,  report_step);
    char * header_file      = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ecl_summary_header_file , fmt_file , -1);
    summary = ecl_sum_fread_alloc(header_file , 1 , (const char **) &summary_file , true , enkf_config_get_endian_swap(enkf_state->config));
    free(summary_file);
    free(header_file);
  } 
  
  /*****************************************************************/
  
  
  /**
     Iterating through the restart block:
     
      1. Build up enkf_state->restart_kw_list.
      2. Send static keywords straight out.
  */
  
  {
    restart_kw_list_type * block_kw_list = ecl_block_get_restart_kw_list(restart_block);
    const char * block_kw 	         = restart_kw_list_get_first(block_kw_list);
    
    restart_kw_list_reset(enkf_state->restart_kw_list);
    while (block_kw != NULL) {
      char * kw = util_alloc_string_copy(block_kw);
      enkf_impl_type impl_type;
      ecl_util_escape_kw(kw);
      
      if (enkf_config_has_key(enkf_state->config , kw)) {
	const enkf_config_node_type * config_node = enkf_config_get_node_ref(enkf_state->config , kw);
	impl_type = enkf_config_node_get_impl_type(config_node);
      } else
	impl_type = STATIC;
      

      if (impl_type == FIELD) 
	restart_kw_list_add(enkf_state->restart_kw_list , kw);
      else if (impl_type == STATIC) {
	/* It is a static kw like INTEHEAD or SCON */
	if (enkf_config_include_static_kw(enkf_state->config , kw)) {
	  restart_kw_list_add(enkf_state->restart_kw_list , kw);
	  if (!enkf_config_has_key(enkf_state->config , kw)) 
	    enkf_config_add_type(enkf_state->config , kw , ecl_static , STATIC , NULL , NULL , NULL);
	  
	  if (!enkf_state_has_node(enkf_state , kw)) {
	    const enkf_config_node_type * config_node = enkf_config_get_node_ref(enkf_state->config , kw);
	    enkf_state_add_node(enkf_state , kw , config_node); 
	  }
	  
	  {
	    enkf_node_type * enkf_node         = enkf_state_get_node(enkf_state , kw);
	    ecl_static_kw_type * ecl_static_kw = enkf_node_value_ptr(enkf_node);
	    ecl_static_kw_inc_counter(ecl_static_kw , true , report_step);
	    enkf_node_ecl_load_static(enkf_node , ecl_block_iget_kw(restart_block , block_kw , ecl_static_kw_get_counter( ecl_static_kw )) , report_step);
	    /*
	      Static kewyords go straight out ....
	    */
	    enkf_fs_fwrite_node(shared_info->fs , enkf_node , report_step , my_config->iens , forecast);
	    enkf_node_free_data(enkf_node);
	  }
	} 
      } else
	util_abort("%s: hm - something wrong - can (currently) only load FIELD/STATIC implementations from restart files - aborting \n",__func__);
      
      free(kw);
      block_kw = restart_kw_list_get_next(block_kw_list);
    }
  }
  
  /******************************************************************/
  /** Starting on the enkf_node_ecl_load() function calls. This is
      where the actual loading (apart from static keywords) is done.*/
  
  {
    const int num_keys = hash_get_size(enkf_state->node_hash);
    char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
    int ikey;
    for (ikey= 0; ikey < num_keys; ikey++) {
      enkf_node_type *enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
      if (enkf_node_has_func(enkf_node , ecl_load_func))
	if (enkf_node_include_type(enkf_node , mask))
	  enkf_node_ecl_load(enkf_node , run_info->run_path , my_config->eclbase , summary , restart_block , report_step);
      
    }                                                                      
    util_free_stringlist(key_list , num_keys);
  }
  

  /*****************************************************************/
  /* Cleaning up */

  if (summary != NULL) ecl_sum_free( summary );
  if (restart_block != NULL) ecl_block_free( restart_block );
}



/**
   The actual loading is done by the function enkf_state_ecl_load2().
*/
void enkf_state_ecl_load(enkf_state_type * enkf_state , enkf_obs_type * enkf_obs , bool unified , int report_step1 , int report_step2) {
  /*enkf_state_ecl_store(enkf_state , report_step1 , report_step2);*/
  
  
  /*
    Loading in the X0000 files containing the initial distribution of
    pressure/saturations/....
  */

  if (report_step1 == 0) {
    enkf_state_ecl_load2(enkf_state , unified , ecl_restart , 0);
    enkf_state_fwrite(enkf_state , ecl_restart , 0 , analyzed);
  }
  enkf_state_ecl_load2(enkf_state , unified , ecl_restart + ecl_summary , report_step2);
  
  /* Burde ha et eget measure flag */
  enkf_state_fwrite(enkf_state , ecl_restart + ecl_summary , report_step2 , forecast);
  enkf_state_measure(enkf_state , enkf_obs);  
}



/**
   Observe that this function uses run_info->step1 to load all the nodes which
   are needed in the restart file. I.e. if you have carefully prepared a funny
   state with dynamic/static data which do not agree with the current value of
   run_info->step1 YOUR STATE WILL BE OVERWRITTEN.
*/

static void enkf_state_write_restart_file(enkf_state_type * enkf_state) {
  shared_info_type * shared_info       = enkf_state->shared_info;
  const member_config_type * my_config = enkf_state->my_config;
  const run_info_type      * run_info  = enkf_state->run_info;
  bool endian_swap       = enkf_config_get_endian_swap(enkf_state->config);
  const bool fmt_file    = enkf_state_fmt_file(enkf_state);
  char * restart_file    = ecl_util_alloc_filename(run_info->run_path , my_config->eclbase , ecl_restart_file , fmt_file , run_info->step1);
  fortio_type * fortio   = fortio_fopen(restart_file , "w" , endian_swap);
  const char * kw;

  if (restart_kw_list_empty(enkf_state->restart_kw_list))
    enkf_fs_fread_restart_kw_list(shared_info->fs , run_info->step1 , my_config->iens , enkf_state->restart_kw_list);
  

  kw = restart_kw_list_get_first(enkf_state->restart_kw_list);
  while (kw != NULL) {
    /* 
       If the restart kw_list asks for a keyword which we do not have,
       we assume it is a static keyword and add it it to the
       enkf_state instance. 
       
       This is a bit unfortunate, because a bug/problem of some sort,
       might be masked (seemingly solved) by adding a static keyword,
       before things blow up completely at a later instant.
    */
    
    if (!enkf_config_has_key(enkf_state->config , kw)) 
      enkf_config_add_type(enkf_state->config , kw , ecl_static , STATIC , NULL , NULL , NULL);
    
    if (!enkf_state_has_node(enkf_state , kw)) {
      const enkf_config_node_type * config_node = enkf_config_get_node_ref(enkf_state->config , kw);
      enkf_state_add_node(enkf_state , kw , config_node); 
    }
	
    {
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw); 
      enkf_var_type var_type = enkf_node_get_var_type(enkf_node); 
      if (var_type == ecl_static) 
	ecl_static_kw_inc_counter(enkf_node_value_ptr(enkf_node) , false , run_info->step1);
      enkf_fs_fread_node(shared_info->fs , enkf_node , run_info->step1 , my_config->iens , run_info->init_state);
      
      if (var_type == ecl_restart) {
	/* Pressure and saturations */
	if (enkf_node_get_impl_type(enkf_node) == FIELD)
	  enkf_node_ecl_write_fortio(enkf_node , fortio , fmt_file , FIELD);
	else 
	  util_abort("%s: internal error wrong implementetion type:%d - node:%s aborting \n",__func__ , enkf_node_get_impl_type(enkf_node) , enkf_node_get_key_ref(enkf_node));
      } else if (var_type == ecl_static) {
	enkf_node_ecl_write_fortio(enkf_node , fortio , fmt_file , STATIC );
	enkf_node_free_data(enkf_node); /* Just immediately discard the static data. */
      } else 
	util_abort("%s: internal error - should not be here ... \n",__func__);
    }
    kw = restart_kw_list_get_next(enkf_state->restart_kw_list);
  }
  fortio_fclose(fortio);
}



/**
  This function writes out all the files needed by an ECLIPSE simulation, this
  includes the restart file, and the various INCLUDE files corresponding to
  parameteres estimated by EnKF.

  The writing of restart file is delegated to enkf_state_write_restart_file().
*/

void enkf_state_ecl_write(enkf_state_type * enkf_state ,  int mask) {
  const run_info_type * run_info         = enkf_state->run_info;
  int    restart_mask    = 0;

  if (mask & ecl_restart) 
    restart_mask += ecl_restart;
  if (mask & ecl_static)
    restart_mask += ecl_static;
  mask -= restart_mask;

  if (restart_mask > 0 && run_info->step1 > 0)
    enkf_state_write_restart_file(enkf_state);
  
  util_make_path(run_info->run_path);
  {
    const int num_keys = hash_get_size(enkf_state->node_hash);
    char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
    int ikey;
    
    for (ikey = 0; ikey < num_keys; ikey++) {
      enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
      if (enkf_node_include_type(enkf_node , mask)) 
	enkf_node_ecl_write(enkf_node , run_info->run_path);
    }
    util_free_stringlist(key_list , num_keys);
  }
}


/**
   This function takes a report_step and a analyzed|forecast state as
  input; the enkf_state instance is set accordingly and written to
  disk.
*/

void enkf_state_fwrite(const enkf_state_type * enkf_state , int mask , int report_step , state_enum state) {
  shared_info_type * shared_info = enkf_state->shared_info;
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;
  
  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_fs_fwrite_node(shared_info->fs , enkf_node , report_step , my_config->iens , state);
  }                                                                     
  util_free_stringlist(key_list , num_keys);
}


void enkf_state_fread(enkf_state_type * enkf_state , int mask , int report_step , state_enum state) {
  shared_info_type * shared_info = enkf_state->shared_info;
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;
  
  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_fs_fread_node(shared_info->fs , enkf_node , report_step , my_config->iens , state);
  }                                                                     
  util_free_stringlist(key_list , num_keys);
}


void enkf_state_free_nodes(enkf_state_type * enkf_state, int mask) {
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;
  
  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_include_type(enkf_node , mask)) 
      enkf_state_del_node(enkf_state , enkf_node_get_key_ref(enkf_node));
  }                                                                     
  util_free_stringlist(key_list , num_keys);
}

      


meas_vector_type * enkf_state_get_meas_vector(const enkf_state_type *state) {
  return state->meas_vector;
}


void enkf_state_free(enkf_state_type *enkf_state) {
  hash_free(enkf_state->node_hash);
  hash_free(enkf_state->data_kw);
  restart_kw_list_free(enkf_state->restart_kw_list);

  member_config_free(enkf_state->my_config);
  run_info_free(enkf_state->run_info);
  shared_info_free(enkf_state->shared_info);

  free(enkf_state);
}



enkf_node_type * enkf_state_get_node(const enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , node_key);
    return enkf_node;
  } else {
    util_abort("%s: node:%s not found in state object - aborting \n",__func__ , node_key);
    return NULL; /* Compiler shut up */
  }
}



void enkf_state_del_node(enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) 
    hash_del(enkf_state->node_hash , node_key);
  else 
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
   report_step1 : The simulation should start from this report step; 
                  dynamic data are loaded from this step.
   report_step2 : The simulation should stop at this report step.

   For a normal EnKF run we well have init_step == report_step1, but
   in the case where we want rerun from the beginning with updated
   parameters, they will be different. If init_step != report_step1,
   it is required that report_step1 == 0; otherwise the dynamic data
   will become completely inconsistent. We just don't allow that!
*/


void enkf_state_init_eclipse(enkf_state_type *enkf_state) {
  const member_config_type  * my_config = enkf_state->my_config;  
  const run_info_type       * run_info  = enkf_state->run_info;
  if (!run_info->can_sim)
    util_abort("%s: this EnKF instance can not start simulations - aborting \n", __func__ );
  {
    const shared_info_type    * shared_info = enkf_state->shared_info;
    const run_info_type       * run_info    = enkf_state->run_info;
    if (!run_info->__ready) 
      util_abort("%s: must initialize run parameters with enkf_state_init_run() first \n",__func__);
    
    
    if (run_info->step1 != run_info->init_step)
      if (run_info->step1 > 0)
	util_abort("%s: internal error - when initializing from a different timestep than starting from - the start step must be zero.\n",__func__);
    
    if (run_info->step1 > 0) {
      char * data_initialize = util_alloc_sprintf("RESTART\n   \'%s\'  %d  /\n" , my_config->eclbase , run_info->step1);
      enkf_state_set_data_kw(enkf_state , "INIT" , data_initialize);
      free(data_initialize);
    }
    
    util_make_path(run_info->run_path);
    {
      char * data_file = ecl_util_alloc_filename(run_info->run_path , my_config->eclbase , ecl_data_file , true , -1);
      util_filter_file(enkf_config_get_data_file(enkf_state->config) , NULL , data_file , '<' , '>' , enkf_state->data_kw , false);
      free(data_file);
    }
    
    {
      char * schedule_file = util_alloc_full_path(run_info->run_path , enkf_config_get_schedule_target_file(enkf_state->config));
      sched_file_fprintf(shared_info->sched_file , run_info->step2 , -1 , -1 , schedule_file);
      free(schedule_file);
    }
    
    /**
       This is a bit tricky:
       
       + Parameters are loaded from the init_step.
       + Dynamic data (and corresponding static) are loaded from step1 - but that is done in the enkf_state_write_restart_file() routine.
    */
    enkf_state_fread(enkf_state , parameter + constant + static_parameter , run_info->init_step , run_info->init_state);
    enkf_state_ecl_write(enkf_state , constant + static_parameter + parameter + ecl_restart + ecl_static);
    
    {
      char * stdin_file = util_alloc_full_path(run_info->run_path , "eclipse.stdin" );  /* The name eclipse.stdin must be mathched when the job is dispatched. */
      ecl_util_init_stdin( stdin_file , my_config->eclbase );
      free(stdin_file);
    }
    
    {
      bool  fmt_file              = enkf_config_get_fmt_file(enkf_state->config);
      hash_type * context    	= hash_alloc();
      char * restart_file1   	= ecl_util_alloc_filename(NULL , my_config->eclbase , ecl_restart_file  	   , fmt_file , run_info->step1);
      char * restart_file2   	= ecl_util_alloc_filename(NULL , my_config->eclbase , ecl_restart_file  	   , fmt_file , run_info->step2);
      char * smspec_file     	= ecl_util_alloc_filename(NULL , my_config->eclbase , ecl_summary_header_file  , fmt_file , -1);
      char * iens            	= util_alloc_sprintf("%d" , my_config->iens);
      char * ecl_base        	= my_config->eclbase;
      char * step1_s  		= util_alloc_sprintf("%d" , run_info->step1);
      char * step2_s  		= util_alloc_sprintf("%d" , run_info->step2);
      
      
      hash_insert_hash_owned_ref( context , "REPORT_STEP1"  , void_arg_alloc_ptr( step1_s ) 	 , void_arg_free__);
      hash_insert_hash_owned_ref( context , "REPORT_STEP2"  , void_arg_alloc_ptr( step2_s ) 	 , void_arg_free__);
      hash_insert_hash_owned_ref( context , "RESTART_FILE1" , void_arg_alloc_ptr( restart_file1 )  , void_arg_free__);
      hash_insert_hash_owned_ref( context , "RESTART_FILE2" , void_arg_alloc_ptr( restart_file2 )  , void_arg_free__);
      hash_insert_hash_owned_ref( context , "SMSPEC_FILE"   , void_arg_alloc_ptr( smspec_file   )  , void_arg_free__);
      hash_insert_hash_owned_ref( context , "ECL_BASE"      , void_arg_alloc_ptr( ecl_base   )     , void_arg_free__);
      hash_insert_hash_owned_ref( context , "IENS"          , void_arg_alloc_ptr( iens   )         , void_arg_free__);
      {
	char ** key_list = hash_alloc_keylist( enkf_state->data_kw );
	int i;
	/*  These are owned by the data_kw hash */
	for (i = 0; i < hash_get_size(enkf_state->data_kw); i++)
	  hash_insert_ref( context , key_list[i] , hash_get( enkf_state->data_kw , key_list[i]));
      }
      ext_joblist_python_fprintf( shared_info->joblist , run_info->forward_model ,run_info->run_path , context);
      
      free(iens);
      free(restart_file1);
      free(restart_file2);
      free(smspec_file);
      free(step1_s);
      free(step2_s);
      hash_free(context);
    }
  }
}






/**
   xx_run_eclipse() has been split in two functions:

   1: enkf_state_start_eclipse()

   2: enkf_state_complete_eclipse()

   Because the first is quite CPU intensive (gunzip), and the number of
   concurrent threads should be limitied. For the second there is one
   thread for each ensemble member. This is handled by the calling scope.
*/



void enkf_state_start_eclipse(enkf_state_type * enkf_state) {
  const run_info_type       * run_info    = enkf_state->run_info;
  if (run_info->active) {  /* if the job is not active we just return .*/
    const shared_info_type    * shared_info = enkf_state->shared_info;
    const member_config_type  * my_config   = enkf_state->my_config;
    
    /*
      Prepare the job and submit it to the queue
    */
    enkf_state_init_eclipse(enkf_state);
    job_queue_add_job(shared_info->job_queue , run_info->run_path , my_config->eclbase , my_config->iens);
  }
}


/** 
    Observe that if run_info == false, this routine will return with
    job_completeOK == true, that might be a bit misleading.
*/

void enkf_state_complete_eclipse(enkf_state_type * enkf_state) {
  run_info_type             * run_info    = enkf_state->run_info;
  run_info->complete_OK = true;
  if (run_info->active) {
    const shared_info_type    * shared_info = enkf_state->shared_info;
    const member_config_type  * my_config   = enkf_state->my_config;
    const int usleep_time = 100000; /* 1/10 of a second */ 
    job_status_type final_status;
    
    
    while (true) {
      final_status = job_queue_export_job_status(shared_info->job_queue , my_config->iens);
      
      if (final_status == job_queue_complete_OK) {
	if (run_info->load_results)
	  enkf_state_ecl_load(enkf_state , shared_info->obs , my_config->unified , run_info->step1 , run_info->step2);
	break;
      } else if (final_status == job_queue_complete_FAIL) {
	fprintf(stderr,"** job:%d failed completely - this will break ... \n",my_config->iens);
	run_info->complete_OK = false;
	break;
      } else usleep(usleep_time);
    } 
    
    /**
       We are about to delete the runpath - but there are many tests:
       
       1. If debug == TRUE we keep the path.
       2. If the integration failed we keep tha path.
       3. If keep_runpath == true we keep the path.
       4. If unlink_run_path == false we keep th path.

       Some confusion/conflict between the last two ..??

    */
    
    if ( enkf_config_get_debug(enkf_state->config) == 0) {
      /* In case the job fails, we leave the run_path directory around for debugging. */
      if (!my_config->keep_runpath)
	if (run_info->unlink_run_path && (final_status == job_queue_complete_OK))
	  util_unlink_path(run_info->run_path);
    }
    run_info_complete_run(enkf_state->run_info);
  }
  run_info->__ready = false;
}



bool enkf_state_run_OK(const enkf_state_type * state) {
  return state->run_info->complete_OK;
}



void * enkf_state_complete_eclipse__(void * __enkf_state) {
  enkf_state_type * enkf_state = enkf_state_safe_cast(__enkf_state);
  enkf_state_complete_eclipse(enkf_state);             
  return NULL ;
}


void * enkf_state_start_eclipse__(void * __enkf_state) {
  enkf_state_type * enkf_state = enkf_state_safe_cast(__enkf_state);
  enkf_state_start_eclipse( enkf_state );
  return NULL ; 
}




/*****************************************************************/

/**

Serial is a matrix:

   x----->
  y
  | 
  |
 \|/  


 I have never been good with columns / rows ....


 Read the documentation about strides in enkf_serialize.c.
*/

void enkf_ensemble_mulX(serial_vector_type * serial_vector , int serial_x_size , int serial_y_size , const double * X , int X_x_stride , int X_y_stride) {
  double * serial_data        = serial_vector_get_data( serial_vector );
  const int   serial_y_stride = serial_vector_get_stride( serial_vector );
  const int   serial_x_stride = 1;
   
  double * line = util_malloc(serial_x_size * sizeof * line , __func__);
  int ix,iy;
  
  for (iy=0; iy < serial_y_size; iy++) {
    if (serial_x_stride == 1) 
      memcpy(line , &serial_data[iy * serial_y_stride] , serial_x_size * sizeof * line);
    else
      for (ix = 0; ix < serial_x_size; ix++)
	line[ix] = serial_data[iy * serial_y_stride + ix * serial_x_stride];

    for (ix = 0; ix < serial_x_size; ix++) {
      int k;
      double dot_product = 0;
      for (k = 0; k < serial_x_size; k++)
	dot_product += line[k] * X[ix * X_x_stride + k*X_y_stride];
      serial_data[ix * serial_x_stride + iy * serial_y_stride] = dot_product;
    }

  }
  
  free(line);
}




/**
   This struct is a helper quantity designed to organize all the information needed by the
   serialize routine. The serializing is done by several threads, each takes one range of
   ensemble members, and get one enkf_update_info_struct with information.
*/

struct enkf_update_info_struct {
  int      iens1;   	   	 	/* The first ensemble member this thread is updating. */
  int      iens2;   	   	 	/* The last ensemble member this thread is updating. */
  serial_vector_type * serial_vector;   /* The holding struct for the serializ vector. */
  char   **key_list;       	 	/* The list of keys in the enkf_state object - shared among all ensemble members. */
  int      num_keys;             	/* The length of the key_list. */ 
  int     *start_ikey;     	 	/* The start index (into key_list) of the current serialization
			   	 	   call. Vector with one element for each ensemble member. */
  int     *next_ikey;      	 	/* The start index of the next serialization call. [RETURN VALUE] */
  size_t  *member_serial_size;   	/* The number of elements serialized pr. member (should be equal for all members, 
                                 	   else something is seriously broken). [RETURN VALUE] */  
  bool    *member_complete;      	/* Boolean pr. member flag - true if the member has been
				 	   completely serialized. [RETURN VALUE] */
  int      update_mask;          	/* An enkf_var_type instance which should be included in the update. */ 
  enkf_state_type ** ensemble;   	/* The actual ensemble. */
  bool     __data_owner;         	/* Whether this instance owns the various pr. ensemble member vectors. */ 
};

typedef struct enkf_update_info_struct enkf_update_info_type;



/**
   This function allocates a vector of enkf_update_info_type
   instances, ONE FOR EACH ACTIVE THREAD. The first element becomes
   the owner of the various allocated resources, the others just point
   to the ones ine the first.

   Observe that since in general we will have many more ensemble
   members than threads, each of these info blocks will serve as info
   containers for many ensemble members.
*/

enkf_update_info_type ** enkf_ensemble_alloc_update_info(enkf_state_type ** ensemble , int ens_size , int update_mask , int num_threads, serial_vector_type * serial_vector) {
  int thread_block_size        = ens_size / num_threads;
  char ** key_list             = hash_alloc_keylist(ensemble[0]->node_hash);
  bool    * member_complete    = util_malloc(ens_size * sizeof * member_complete , __func__);
  int     * start_ikey         = util_malloc(ens_size * sizeof * start_ikey , __func__);
  int     * next_ikey          = util_malloc(ens_size * sizeof * next_ikey , __func__);
  size_t  * member_serial_size = util_malloc(ens_size * sizeof * member_serial_size , __func__);
  {
    enkf_update_info_type ** info_list = util_malloc(num_threads * sizeof * info_list ,__func__);
    int it;
    for (it = 0; it < num_threads; it++) {
      enkf_update_info_type * info = util_malloc(sizeof * info , __func__);
      info->num_keys           = hash_get_size( ensemble[0]->node_hash );
      info->key_list           = key_list;
      info->member_complete    = member_complete;
      info->start_ikey         = start_ikey;
      info->next_ikey          = next_ikey;
      info->member_serial_size = member_serial_size;

      info->serial_vector      = serial_vector; 
      info->update_mask        = update_mask;
      info->ensemble           = ensemble;
      info->iens1              = it * thread_block_size;
      info->iens2              = info->iens1 + thread_block_size; /* Upper limit is *NOT* inclusive */

      if (it == 0)
	info->__data_owner = true;
      else
	info->__data_owner = false;
      info_list[it] = info;
    }
    {
      int iens;
      for (iens = 0; iens < ens_size; iens++) {
	info_list[0]->member_complete[iens] = false;
	info_list[0]->start_ikey[iens]      = 0;
      }
    }
    info_list[num_threads - 1]->iens2 = ens_size;
    return info_list;
  }
}


void enkf_ensemble_free_update_info(enkf_update_info_type ** info_list , int size) {
  int it;
  for (it = 0; it < size; it++) {
    enkf_update_info_type * info = info_list[it];
    if (info->__data_owner) {
      util_free_stringlist(info->key_list , info->num_keys);
      free(info->member_complete);   
      free(info->start_ikey);
      free(info->next_ikey);
      free(info->member_serial_size);
    }
    free(info);
  }
  free(info_list);
}



void * enkf_ensemble_serialize__(void * _info) {
  enkf_update_info_type  * info     = (enkf_update_info_type *) _info;

  int iens1       	      	     = info->iens1;
  int iens2       	      	     = info->iens2;
  serial_vector_type * serial_vector = info->serial_vector;
  enkf_state_type ** ensemble 	     = info->ensemble;
  int update_mask             	     = info->update_mask;
  size_t * member_serial_size 	     = info->member_serial_size;
  bool * member_complete      	     = info->member_complete;
  int * next_ikey  	      	     = info->next_ikey;
  int * start_ikey 	      	     = info->start_ikey;
  char ** key_list            	     = info->key_list;
  int serial_stride                  = serial_vector_get_stride( serial_vector ); 
  int iens;
  
  for (iens = iens1; iens < iens2; iens++) {
    int ikey               = start_ikey[iens];
    bool node_complete     = true;  
    size_t   current_serial_offset = iens;
    enkf_state_type * enkf_state = ensemble[iens];
    
    
    while (node_complete) {                                           
      enkf_node_type *enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
      if (enkf_node_include_type(enkf_node , update_mask)) {                       
	int elements_added            = enkf_node_serialize(enkf_node , current_serial_offset , serial_vector , &node_complete);
	current_serial_offset    += serial_stride * elements_added;  
	member_serial_size[iens] += elements_added;
      }
      
      if (node_complete) {
	ikey += 1;
	if (ikey == hash_get_size(enkf_state->node_hash)) {
	  if (node_complete) member_complete[iens] = true;
	  break;
	}
      }
    }
    /* Restart on this node */
    next_ikey[iens] = ikey;
  }

  return NULL;
}





void enkf_ensemble_update(enkf_state_type ** enkf_ensemble , int ens_size , serial_vector_type * serial_vector , const double * X) {
  const int threads = 1;
  int update_mask   = ecl_summary + ecl_restart + parameter;
  thread_pool_type * tp = thread_pool_alloc(0 /* threads */);
  enkf_update_info_type ** info_list     = enkf_ensemble_alloc_update_info(enkf_ensemble , ens_size , update_mask , threads , serial_vector);
  int       iens , ithread;


  bool state_complete = false;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_apply_NEW2(enkf_ensemble[iens] ,  update_mask , clear_serial_state_func , NULL);

  while (!state_complete) {
    for (iens = 0; iens < ens_size; iens++) 
      info_list[0]->member_serial_size[iens] = 0;  /* Writing only on element[0] - because that is the only member with actual storage. */
    
    for (ithread =  0; ithread < threads; ithread++) 
      thread_pool_add_job(tp , &enkf_ensemble_serialize__ , info_list[ithread]);
    thread_pool_join(tp);

    /*
      This code block is a integrity check - we check that the
      serialization has come equally long with all members. If for
      instance one member is "larger" than the others this test will
      fail.
    */
    {
      bool   * member_complete    = info_list[0]->member_complete;
      size_t * member_serial_size = info_list[0]->member_serial_size;
      for (iens=1; iens < ens_size; iens++) {
	if (member_complete[iens]    != member_complete[iens-1])    util_abort("%s: member_complete difference    - INTERNAL ERROR - aborting \n",__func__); 
	if (member_serial_size[iens] != member_serial_size[iens-1]) util_abort("%s: member_serial_size difference - INTERNAL ERROR - aborting \n",__func__); 
      }
      state_complete = member_complete[0];
    }
    
    {
      enkf_update_info_type * info = info_list[0];

      /* Update section */
      /*enkf_ensemble_mulX(serial_data , 1 , ens_size , ens_size , info->member_serial_size[0] , X , ens_size , 1);*/
      enkf_ensemble_mulX(serial_vector , ens_size , info->member_serial_size[0] , X , ens_size , 1); 



      /* deserialize section */
      for (iens = 0; iens < ens_size; iens++) {
	enkf_state_type * enkf_state = enkf_ensemble[iens];
	int ikey     = info->start_ikey[iens];
	int num_keys = info->num_keys;

	while (1) {
	  enkf_node_type *enkf_node = hash_get(enkf_state->node_hash , info->key_list[ikey]);
	  if (enkf_node_include_type(enkf_node , update_mask)) 
	    enkf_node_deserialize(enkf_node , serial_vector);
	  
	  if (ikey == info->next_ikey[iens])
	    break;
	  
	  ikey++;
	  if (ikey == num_keys)
	    break;
	}
      }
      
      for (iens = 0; iens < ens_size; iens++) 
	info->start_ikey[iens] = info->next_ikey[iens];
    }
  }
  thread_pool_free(tp);
  enkf_ensemble_free_update_info( info_list , threads );
}


void enkf_state_init_run(enkf_state_type * state , bool active , int init_step , state_enum init_state , int step1 , int step2 , bool load_results , bool unlink_run_path , const stringlist_type * forward_model) {
  member_config_type * my_config    = state->my_config;
  shared_info_type   * shared_info  = state->shared_info;

  run_info_set( state->run_info , active , init_step , init_state , step1 , step2 , load_results , unlink_run_path , forward_model , my_config->iens , shared_info->run_path_fmt);
}



