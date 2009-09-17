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
#include <ecl_io_config.h>
#include <ecl_file.h>
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
#include <summary.h>
#include <arg_pack.h>
#include <enkf_fs.h>
#include <basic_driver.h>
#include <node_ctype.h>
#include <job_queue.h>
#include <sched_file.h>
#include <basic_queue_driver.h>
#include <pthread.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <ensemble_config.h>
#include <model_config.h>
#include <site_config.h>
#include <ecl_config.h>
#include <subst.h>
#include <forward_model.h>
#include <log.h>
#include <ecl_endian_flip.h>
#include <ert_template.h>
#define ENKF_STATE_TYPE_ID 78132


/**
   This struct is a pure utility structure used to pack the various
   bits and pieces of information needed to start, monitor, and load
   back results from the forward model simulations. 

   Typcially the values in this struct are set from the enkf_main object
   before a forward_step starts.
*/

typedef struct run_info_struct {
  bool               	  __ready;              /* An attempt to check the internal state - not active yet. */
  bool               	  active;               /* Is this state object active at all - used for instance in ensemble experiments where only some of the members are integrated. */
  bool                    resample_when_fail;   
  int                	  init_step_parameters; /* The report step we initialize parameters from - will often be equal to step1, but can be different. */
  state_enum         	  init_state_parameter; /* Whether we should init from a forecast or an analyzed state - parameters. */
  state_enum              init_state_dynamic;   /* Whether we should init from a forecast or an analyzed state - dynamic state variables. */
  int                     max_internal_submit;  /* How many times the enkf_state object should try to resubmit when the queueu has said everything is OK - but the load fails. */  
  int                     num_internal_submit;
  int                     load_start;           /* When loading back results - start at this step. */
  int                	  step1;                /* The forward model is integrated: step1 -> step2 */
  int                	  step2;  	          
  char            	* run_path;             /* The currently used runpath - is realloced / freed for every step. */
  run_mode_type   	  run_mode;             /* What type of run this is */
  /******************************************************************/
  /* Return value - set by the called routine!!  */
  bool                    runOK;               /* Set to true when the run has completed - AND - the results have been loaded back. */
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
  const model_config_type     * model_config;      /* .... */
  enkf_fs_type                * fs;                /* The filesystem object - used to load and store nodes. */
  ext_joblist_type            * joblist;           /* The list of external jobs which are installed - and *how* they should be run (with Python code) */
  job_queue_type              * job_queue;         /* The queue handling external jobs. (i.e. LSF / rsh / local / ... )*/ 
  log_type                    * logh;              /* The log handle. */
  ert_templates_type          * templates; 
  bool                          statoil_mode;      /* Fucking hack - should be removed. */
} shared_info_type;




/**
   This struct contains information which is private to this
   member. It is initialized at object boot time, and (typically) not
   changed during the simulation. [In principle it could change during
   the simulation, but the current API does not support that.]
*/ 


struct member_config_struct {
  int  		        iens;                /* The ensemble member number of this member. */
  char                * casename;            /* The name of this case - will mosttly be NULL. */
  keep_runpath_type     keep_runpath;        /* Should the run-path directory be left around (for this member)*/
  char 		      * eclbase;             /* The ECLBASE string used for simulations of this member. */
  sched_file_type     * sched_file;          /* The schedule file - can either be a shared pointer to somehwere else - or a pr. member schedule file. */
  bool                  private_sched_file;  /* Is the member config holding a private schedule file - just relevant when freeing up? */ 
};




/*****************************************************************/
/** THE MOST IMPORTANT ENKF OBJECT */

struct enkf_state_struct {
  int                     __id;              	   /* Funny integer used for run_time type checking. */
  stringlist_type       * restart_kw_list;
  hash_type    	   	* node_hash;
  subst_list_type       * subst_list;        	   /* This a list of key - value pairs which are used in a search-replace
                                             	      operation on the ECLIPSE data file. Will at least contain the key INIT"
                                             	      - which will describe initialization of ECLIPSE (EQUIL or RESTART).*/
  hash_type             * subst_description;      /* A table of description of the various subst_list keys. */
  
  const ecl_config_type * ecl_config;        	   /* ecl_config object - pure reference to the enkf_main object. */
  ensemble_config_type  * ensemble_config;   	   /* The config nodes for the enkf_node objects contained in node_hash. */
  
  run_info_type         * run_info;          	   /* Various pieces of information needed by the enkf_state object when running the forward model. Updated for each report step.*/
  shared_info_type      * shared_info;       	   /* Pointers to shared objects which is needed by the enkf_state object (read only). */
  member_config_type    * my_config;         	   /* Private config information for this member; not updated during a simulation. */
  
  forward_model_type    * default_forward_model;   /* The forward model - owned by this enkf_state instance. */
  forward_model_type    * special_forward_model;   /* A special forward model for a slected report step - explicitly set with enkf_state_set_special_forward_model. */
  forward_model_type    * forward_model;           /* */
};

/*****************************************************************/


static void run_info_set_run_path(run_info_type * run_info , int iens , path_fmt_type * run_path_fmt, const subst_list_type * state_subst_list) {
  util_safe_free(run_info->run_path);
  {
    char * tmp = path_fmt_alloc_path(run_path_fmt , false , iens , run_info->step1 , run_info->step2);
    run_info->run_path = subst_list_alloc_filtered_string( state_subst_list , tmp );
    free( tmp );
  }
  util_make_path( run_info->run_path );
}



/**
   This function sets the run_info parameters. This is typically called
   (via an enkf_state__ routine) by the external scope handling the forward model.

   When this initialization code has been run we are certain that the
   enkf_state object has all the information it needs to "run itself"
   forward.
*/



static void run_info_summarize( const run_info_type * run_info ) {
  printf("Activ.....................: %d \n",run_info->active);
  printf("Loading parameters from...: %d \n",run_info->init_step_parameters);
  printf("Loading state from........: %d \n",run_info->step1);
  printf("Simulating to.............: %d \n",run_info->step2);
  printf("Loading from step.........: %d \n\n",run_info->load_start);
}


static void run_info_set(run_info_type * run_info        , 
			 run_mode_type run_mode          , 
			 bool active                     , 
                         bool resample_when_fail         ,
                         int max_internal_submit         ,
			 int init_step_parameters        ,      
			 state_enum init_state_parameter ,
                         state_enum init_state_dynamic   ,
                         int load_start                  , 
			 int step1                       , 
			 int step2                       ,      
			 forward_model_type * __forward_model , 
			 int iens                             , 
			 path_fmt_type * run_path_fmt ,
                         const subst_list_type * state_subst_list) {

  run_info->active               = active;
  run_info->init_step_parameters = init_step_parameters;
  run_info->init_state_parameter = init_state_parameter;
  run_info->init_state_dynamic = init_state_dynamic;
  run_info->step1      	         = step1;
  run_info->step2      	         = step2;
  run_info->runOK                = false;
  run_info->__ready              = true;
  run_info->run_mode             = run_mode;
  run_info->load_start           = load_start;
  run_info->resample_when_fail   = resample_when_fail;
  run_info->max_internal_submit  = max_internal_submit;
  run_info->num_internal_submit  = 0;
  run_info_set_run_path(run_info , iens , run_path_fmt , state_subst_list );
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
  if (run_info->runOK)
    run_info->run_path = util_safe_free(run_info->run_path);
}



/*****************************************************************/

static shared_info_type * shared_info_alloc(const site_config_type * site_config , const model_config_type * model_config, enkf_fs_type * fs , log_type * logh , ert_templates_type * templates) {
  shared_info_type * shared_info = util_malloc(sizeof * shared_info , __func__);

  shared_info->fs           = fs;
  shared_info->joblist      = site_config_get_installed_jobs( site_config );
  shared_info->job_queue    = site_config_get_job_queue( site_config );
  shared_info->model_config = model_config;
  shared_info->statoil_mode = site_config_get_statoil_mode( site_config );
  shared_info->logh         = logh;
  shared_info->templates    = templates;
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



static void member_config_set_eclbase(member_config_type * member_config , const ecl_config_type * ecl_config , const subst_list_type * subst_list) {
  util_safe_free( member_config->eclbase );
  {
    char * tmp = path_fmt_alloc_path(ecl_config_get_eclbase_fmt(ecl_config), false , member_config->iens);
    member_config->eclbase = subst_list_alloc_filtered_string( subst_list , tmp );
    free( tmp );
  }
}


static int member_config_get_last_restart_nr( const member_config_type * member_config) {
  return sched_file_get_num_restart_files( member_config->sched_file ) - 1; /* Fuck me +/- 1 */
}
				   

static void member_config_free(member_config_type * member_config) {
  util_safe_free(member_config->eclbase);
  util_safe_free(member_config->casename );

  if (member_config->private_sched_file)
    sched_file_free( member_config->sched_file );
  
  free(member_config);
}



static void member_config_set_keep_runpath(member_config_type * member_config , keep_runpath_type keep_runpath) {
  member_config->keep_runpath   = keep_runpath;
}



static member_config_type * member_config_alloc(int iens , 
                                                const char * casename , 
						keep_runpath_type            keep_runpath , 
						const ecl_config_type      * ecl_config , 
						const ensemble_config_type * ensemble_config) {
						
  member_config_type * member_config = util_malloc(sizeof * member_config , __func__);
  member_config->casename            = util_alloc_string_copy( casename );
  member_config->iens                = iens; /* Can only be changed in the allocater. */
  member_config->eclbase  	     = NULL;
  member_config_set_keep_runpath(member_config , keep_runpath);
  {
    char * schedule_prediction_file = ecl_config_alloc_schedule_prediction_file(ecl_config , iens);
    if (schedule_prediction_file != NULL) {
      member_config->sched_file = sched_file_alloc_copy( ecl_config_get_sched_file( ecl_config ) , false); /* The historic part is a shallow copy. */
      sched_file_parse_append( member_config->sched_file , schedule_prediction_file );
      member_config->private_sched_file = true;
      free( schedule_prediction_file );
    } else {
      member_config->sched_file         = ecl_config_get_sched_file( ecl_config );
      member_config->private_sched_file = false;
    }
  }
  return member_config;
}



static const sched_file_type * member_config_get_sched_file( const member_config_type * member_config) {
  return member_config->sched_file;
}

keep_runpath_type member_config_get_keep_runpath(const member_config_type * member_config) {
	return member_config->keep_runpath;
}


/*****************************************************************/
/** Helper classes complete - starting on the enkf_state proper object. */
/*****************************************************************/



void enkf_state_initialize(enkf_state_type * enkf_state , const stringlist_type * param_list) {
  int ip;
  for (ip = 0; ip < stringlist_get_size(param_list); ip++) {
    int iens = enkf_state_get_iens( enkf_state );
    enkf_node_type * param_node = enkf_state_get_node( enkf_state , stringlist_iget( param_list , ip));
    if (enkf_node_initialize( param_node , iens))
      enkf_fs_fwrite_node(enkf_state_get_fs_ref( enkf_state ) , param_node , 0 , iens , ANALYZED);
  }
}


void enkf_state_init_forward_model(enkf_state_type * enkf_state) {
  member_config_type * member_config = enkf_state->my_config;
  char * iens_s       	  = util_alloc_sprintf("%d"   , member_config->iens);
  char * iens4_s      	  = util_alloc_sprintf("%04d" , member_config->iens);
  char * smspec_file  	  = ecl_util_alloc_filename(NULL , member_config->eclbase , ECL_SUMMARY_HEADER_FILE , ecl_config_get_formatted(enkf_state->ecl_config) , -1);
  char * cwd              = util_alloc_cwd();
  
  free(cwd);
  free(iens_s);
  free(iens4_s);
  free(smspec_file);
}



void enkf_state_set_special_forward_model(enkf_state_type * enkf_state , forward_model_type * forward_model) {
  enkf_state->special_forward_model = forward_model_alloc_copy( forward_model , enkf_state->shared_info->statoil_mode);  /* Discarded again at the end of this run */
  enkf_state->forward_model = enkf_state->special_forward_model;
  enkf_state_init_forward_model(enkf_state);
}



void enkf_state_apply_NEW2(enkf_state_type * enkf_state , int mask , node_function_type function_type, arg_pack_type * arg) {
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




/*
  void enkf_state_set_iens(enkf_state_type * enkf_state , int iens) {
  enkf_state->my_iens = iens;
  }
*/

int  enkf_state_get_iens(const enkf_state_type * enkf_state) {
  return enkf_state->my_config->iens;
}

member_config_type *enkf_state_get_member_config(const enkf_state_type * enkf_state) {
	return enkf_state->my_config;
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



static void enkf_state_add_subst_kw(enkf_state_type * enkf_state , const char * kw , const char * value , const char * description) {
  char * tagged_key = enkf_util_alloc_tagged_string( kw );
  subst_list_insert_owned_ref(enkf_state->subst_list , tagged_key , util_alloc_string_copy(value));
  if (description != NULL) 
    hash_insert_hash_owned_ref( enkf_state->subst_description , tagged_key , util_alloc_string_copy( description ) , free);
  else {
    if (!hash_has_key( enkf_state->subst_description , tagged_key))
      util_abort("%s: must provide documentation for subst key:%s \n",__func__ , kw);
  }
    
  free(tagged_key);
}



/**
   With three DATES keywords in the SCHEDULE file this will
   return "2".
*/

int enkf_state_get_last_restart_nr(const enkf_state_type * enkf_state ) {
  return member_config_get_last_restart_nr( enkf_state->my_config );
}



/**
   Sets all the static subst keywords which will not change during the simulation.
*/
static void enkf_state_set_static_subst_kw(enkf_state_type * enkf_state) {
  {
    char * iens_s      = util_alloc_sprintf("%d"   , enkf_state->my_config->iens);
    char * iens4_s     = util_alloc_sprintf("%04d" , enkf_state->my_config->iens);
    char * cwd         = util_alloc_cwd();
    
    enkf_state_add_subst_kw(enkf_state , "CONFIG_PATH" , cwd     , NULL);  /* Alias for CWD */
    enkf_state_add_subst_kw(enkf_state , "CWD"         , cwd     , NULL); 
    enkf_state_add_subst_kw(enkf_state , "IENS"        , iens_s  , NULL);
    enkf_state_add_subst_kw(enkf_state , "IENS4"       , iens4_s , NULL);
    
    free(cwd);
    free(iens_s);
    free(iens4_s);
  }
  
  {
    member_config_type * my_config = enkf_state->my_config;
    member_config_set_eclbase(my_config , enkf_state->ecl_config , enkf_state->subst_list);
  
    {
      char * smspec_file = ecl_util_alloc_filename(NULL  , my_config->eclbase , ECL_SUMMARY_HEADER_FILE , ecl_config_get_formatted(enkf_state->ecl_config) , -1);
      enkf_state_add_subst_kw(enkf_state , "ECL_BASE"    , my_config->eclbase , NULL);  /* Can not change run_time .. */
      enkf_state_add_subst_kw(enkf_state , "ECLBASE"     , my_config->eclbase , NULL);  /* Can not change run_time .. */
      enkf_state_add_subst_kw(enkf_state , "SMSPEC"      , smspec_file , NULL);
      if (my_config->casename != NULL)
        enkf_state_add_subst_kw( enkf_state , "CASE" , my_config->eclbase , NULL);  /* No CASE_TABLE loaded - using the eclbase as default. */
      else
        enkf_state_add_subst_kw( enkf_state , "CASE" , my_config->casename , NULL);
      
      free(smspec_file);
    }
  }
}

/**
   Two small callback functions used to return (string representation)
   of random integer and random float. The memorty allocated by these
   functions will be freed by the calling scope.
*/

static char * enkf_state_subst_randint(const char * key , void * arg) {
  return util_alloc_sprintf("%d" , rand());
}

static char * enkf_state_subst_randfloat(const char * key , void * arg) {
  return util_alloc_sprintf("%12.10f" , 1.0 * rand() / RAND_MAX);
}

enkf_state_type * enkf_state_alloc(int iens,
                                   enkf_fs_type              * fs, 
                                   const char                * casename , 
				   keep_runpath_type           keep_runpath , 
				   const model_config_type   * model_config,
				   ensemble_config_type      * ensemble_config,
				   const site_config_type    * site_config,
				   const ecl_config_type     * ecl_config,
				   hash_type                 * data_kw,
				   const forward_model_type  * default_forward_model,
                                   log_type                  * logh,
                                   ert_templates_type        * templates) {
  
  enkf_state_type * enkf_state  = util_malloc(sizeof *enkf_state , __func__);
  enkf_state->__id              = ENKF_STATE_TYPE_ID; 

  enkf_state->ensemble_config   = ensemble_config;
  enkf_state->ecl_config        = ecl_config;
  enkf_state->shared_info       = shared_info_alloc(site_config , model_config , fs , logh, templates);
  enkf_state->run_info          = run_info_alloc();
  
  enkf_state->node_hash         = hash_alloc();
  enkf_state->restart_kw_list   = stringlist_alloc_new();

  enkf_state->subst_list        = subst_list_alloc();
  enkf_state->subst_description = hash_alloc();
  
  /*
    The user MUST specify an INIT_FILE, and for the first timestep the
   <INIT> tag in the data file will be replaced by an 

INCLDUE
   EQUIL_INIT_FILE

   statement. When considering the possibility of estimating EQUIL this
   require a real understanding of the treatment of paths:

   * If not estimating the EQUIL contacts, all members should use the
     same init_file. To ensure this the user must specify the ABSOLUTE
     PATH to a file containing initialization information.

   * If the user is estimating initial contacts, the INIT_FILE must
     point to the ecl_file of the EQUIL keyword, this must be a pure
     filename without any path component (as it will be generated by
     the EnKF program, and placed in the run_path directory). We could
     let the EnKF program use the ecl_file of the EQUIL keyword if it
     is present.

   The <INIT> key is actually initialized in the
   enkf_state_set_dynamic_subst_kw() function.
  */

  /**
     Adding all the subst_kw keywords here, with description. Listing
     all of them here in one go guarantees that we have control over
     the ordering (which is interesting because the substititions are
     done in cacade like fashion).
  */
  enkf_state_add_subst_kw(enkf_state , "RUNPATH"       , "---" , "The absolute path of the current forward model instance. ");
  enkf_state_add_subst_kw(enkf_state , "CONFIG_PATH"   , "---" , "The working directory of the enkf simulation == the location of the configuration file.");
  enkf_state_add_subst_kw(enkf_state , "CWD"           , "---" , "The working directory of the enkf simulation == the location of the configuration file.");
  enkf_state_add_subst_kw(enkf_state , "IENS"          , "---" , "The realisation number for this realization.");
  enkf_state_add_subst_kw(enkf_state , "IENS4"         , "---" , "The realization number for this realization - formated with %04d.");
  enkf_state_add_subst_kw(enkf_state , "ECLBASE"       , "---" , "The ECLIPSE basename for this realization.");
  enkf_state_add_subst_kw(enkf_state , "ECL_BASE"      , "---" , "Depreceated - use ECLBASE instead.");
  enkf_state_add_subst_kw(enkf_state , "INIT"          , "---" , "The string which will be inserted instead of <INIT> in the ECLIPSE data file.");
  enkf_state_add_subst_kw(enkf_state , "SMSPEC"        , "---" , "The ECLIPSE SMSPEC file for this realization.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP1"        , "---" , "The initial report step for this simulation.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP2"        , "---" , "The final report step for this simulation.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP1_04"     , "---" , "The initial report step for this simulation - formated with %04d.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP2_04"     , "---" , "The final report step for this simulation - formated withh %04d.");
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE1" , "---" , "The ECLIPSE restart file this simulation starts with.");
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE2" , "---" , "The ECLIPSE restart file this simulation should end with.");
  enkf_state_add_subst_kw(enkf_state , "RANDINT"       , NULL  , "Random integer value");
  enkf_state_add_subst_kw(enkf_state , "RANDFLOAT"     , NULL  , "Random float value");
  if (casename != NULL) 
    enkf_state_add_subst_kw(enkf_state , "CASE" , casename , "The casename for this realization - as loaded from the CASE_TABLE file.");
  else
    enkf_state_add_subst_kw(enkf_state , "CASE" , "---" , "The casename for this realization - similar to ECLBASE.");
  
  {
    /** Adding substitute callbacks */
    char * tagged_randint   = enkf_util_alloc_tagged_string( "RANDINT" );
    char * tagged_randfloat = enkf_util_alloc_tagged_string( "RANDFLOAT" );
    subst_list_insert_callback( enkf_state->subst_list , tagged_randint   , enkf_state_subst_randint   , NULL , NULL);
    subst_list_insert_callback( enkf_state->subst_list , tagged_randfloat , enkf_state_subst_randfloat , NULL , NULL);
    free( tagged_randint );
    free( tagged_randfloat );
  }

  {
    hash_iter_type * iter = hash_iter_alloc(data_kw);
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      enkf_state_add_subst_kw(enkf_state , key , hash_get(data_kw , key) , "Supplied by the user in the enkf configuration file");
      key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
  }
  enkf_state->my_config = member_config_alloc( iens , casename , keep_runpath , ecl_config , ensemble_config);
  enkf_state_set_static_subst_kw(  enkf_state );


  enkf_state->special_forward_model = NULL;
  enkf_state->default_forward_model = forward_model_alloc_copy( default_forward_model , site_config_get_statoil_mode(site_config));
  enkf_state->forward_model         = enkf_state->default_forward_model;
  enkf_state_init_forward_model( enkf_state );
  
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


void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_key , const enkf_config_node_type * config) {
  enkf_node_type *enkf_node = enkf_node_alloc(config);
  if (enkf_state_has_node(enkf_state , node_key)) 
    util_abort("%s: node:%s already added  - aborting \n",__func__ , node_key);
  hash_insert_hash_owned_ref(enkf_state->node_hash , node_key , enkf_node, enkf_node_free__);
  
  /* Setting the global subst list so that the GEN_KW templates can contain e.g. <IENS> and <CWD>. */
  if (enkf_node_get_impl_type( enkf_node ) == GEN_KW)
    gen_kw_set_global_subst_list( enkf_node_value_ptr( enkf_node ) , enkf_state->subst_list );
}



/**
   This function loads the dynamic results from one report step. In
   ECLIPSE speak this implies loading the summary data. Observe the following:

    2. When loading results from several report_steps consecutively
       the whole thing could be speeded up by doing it one go. That is
       currently not implemented.

   The results are immediately stored in the enkf_fs filesystem.
*/
  
/* 
   When we arrive in this function we *KNOW* that summary should be
   loaded from disk.
*/
static void enkf_state_internalize_dynamic_results(enkf_state_type * enkf_state , const model_config_type * model_config , int report_step, bool * loadOK) {
  /* IFF reservoir_simulator == ECLIPSE ... */
  if (report_step > 0) {
    const shared_info_type   * shared_info = enkf_state->shared_info;
    const member_config_type * my_config   = enkf_state->my_config;
    const run_info_type   * run_info       = enkf_state->run_info;
    const ecl_config_type * ecl_config     = enkf_state->ecl_config;
    const bool fmt_file  		   = ecl_config_get_formatted(ecl_config);
    const bool internalize_all             = model_config_internalize_results( model_config , report_step );
    const int  iens                        = my_config->iens;
    
    char * summary_file     = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ECL_SUMMARY_FILE        , fmt_file ,  report_step);
    char * header_file      = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ECL_SUMMARY_HEADER_FILE , fmt_file , -1);
    ecl_sum_type * summary;

    if ((summary_file != NULL) && (header_file != NULL)) {
      summary = ecl_sum_fread_alloc(header_file , 1 , (const char **) &summary_file );
      free( summary_file );
      free( header_file );
    } else
      summary = NULL;  /* OK - no summary data was found on the disk. */
    
    /* The actual loading */
    {
      hash_iter_type * iter = hash_iter_alloc(enkf_state->node_hash);
      while ( !hash_iter_is_complete(iter) ) {
        enkf_node_type * node = hash_iter_get_next_value(iter);
	if (enkf_node_get_var_type(node) == DYNAMIC_RESULT) {
	  bool internalize = internalize_all;
	  if (!internalize)  /* If we are not set up to load the full state - then we query this particular node. */
	    internalize = enkf_node_internalize(node , report_step);

	  if (internalize) {
	    if (enkf_node_ecl_load(node , run_info->run_path , summary , NULL , report_step , iens))  /* Loading/internalizing */
              enkf_fs_fwrite_node(shared_info->fs , node , report_step , iens , FORECAST);            /* Saving to disk */
            else {
              *loadOK = false;
              log_add_fmt_message(shared_info->logh , 3 , "[%03d:%04d] Failed load data for node:%s.",my_config->iens , report_step , enkf_node_get_key( node ));
            }
	  } 
	}
      }
      hash_iter_free(iter);
    }
    
    if (summary != NULL)
      ecl_sum_free( summary );
  }
}



/**
   The ECLIPSE restart files can contain several instances of the same
   keyword, e.g. AQUIFER info can come several times with identical
   headers, also when LGR is in use the same header for
   e.g. PRESSURE/SWAT/INTEHEAD/... wilkl occur several times. The
   enkf_state/ensembl_config objects require unique keys.

   This function takes keyword string and an occurence number, and
   combine them to one new string like this:

     __realloc_static_kw("INTEHEAD" , 0) ==>  "INTEHEAD_0"

   In the enkf layer the key used will then be INTEHEAD_0. 
*/


static char * __realloc_static_kw(char * kw , int occurence) {
  char * new_kw = util_alloc_sprintf("%s_%d" , kw , occurence);
  free(kw);
  ecl_util_escape_kw(new_kw);  
  return new_kw;
}



/**
   This function loads the STATE from a forward simulation. In ECLIPSE
   speak that means to load the solution vectors (PRESSURE/SWAT/..)
   and the necessary static keywords.
   
   When the state has been loaded it goes straight to disk.
*/

static void enkf_state_internalize_state(enkf_state_type * enkf_state , const model_config_type * model_config , int report_step , bool * loadOK) {
  member_config_type * my_config   = enkf_state->my_config;
  shared_info_type   * shared_info = enkf_state->shared_info;
  run_info_type      * run_info    = enkf_state->run_info;
  const bool fmt_file              = ecl_config_get_formatted(enkf_state->ecl_config);
  const bool unified               = ecl_config_get_unified(enkf_state->ecl_config);
  const bool internalize_state     = model_config_internalize_state( model_config , report_step );
  ecl_file_type  * restart_file;
  
  
  /**
     Loading the restart block.
  */

  if (unified) 
    util_abort("%s: sorry - unified restart files are not supported \n",__func__);
  {
    char * file  = ecl_util_alloc_exfilename(run_info->run_path , my_config->eclbase , ECL_RESTART_FILE , fmt_file , report_step);
    if (file != NULL) {
      restart_file = ecl_file_fread_alloc(file );
      free(file);
    } else 
      restart_file = NULL;  /* No restart information was found; if that is expected the program will fail hard in the enkf_node_ecl_load() functions. */
  }
  
  /*****************************************************************/
  
  
  /**
     Iterating through the restart file:
     
     1. Build up enkf_state->restart_kw_list.
     2. Send static keywords straight out.
  */
  
  if (restart_file != NULL) {
    stringlist_clear( enkf_state->restart_kw_list );
    {
      int ikw; 
      for (ikw =0; ikw < ecl_file_get_num_kw( restart_file ); ikw++) {
	enkf_impl_type impl_type;
	const ecl_kw_type * ecl_kw = ecl_file_iget_kw( restart_file , ikw);
	int occurence              = ecl_file_iget_occurence( restart_file , ikw ); /* This is essentially the static counter value. */
	char * kw                  = ecl_kw_alloc_strip_header( ecl_kw );
	/** 
	    Observe that this test will never succeed for static keywords,
	    because the internalized key has appended a _<occurence>.
	*/
	if (ensemble_config_has_key(enkf_state->ensemble_config , kw)) {
	  /**
	     This is poor-mans treatment of LGR. When LGR is used the restart file
	     will contain repeated occurences of solution vectors, like
	     PRESSURE. The first occurence of PRESSURE will be for the ordinary
	     grid, and then there will be subsequent PRESSURE sections for each
	     LGR section. The way this is implemented here is as follows:
	     
	     1. The first occurence of pressure is internalized as the enkf_node
	        pressure (if we indeed have a pressure node).
	     
	     2. The consecutive pressure nodes are internalized as static
	        parameters.
	   
	     The variable 'occurence' is the key here.
	  */
	  
	  if (occurence == 0) {
	    const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_state->ensemble_config , kw);
	    impl_type = enkf_config_node_get_impl_type(config_node);
	  } else 
	    impl_type = STATIC;
	} else
	  impl_type = STATIC;
	
	
	if (impl_type == FIELD) 
	  stringlist_append_copy(enkf_state->restart_kw_list , kw);
	else if (impl_type == STATIC) {
	  if (ecl_config_include_static_kw(enkf_state->ecl_config , kw)) {
	    /* It is a static kw like INTEHEAD or SCON */
	    /* 
	       Observe that for static keywords we do NOT ask the node 'privately' if
	       internalize_state is false: It is impossible to single out static keywords for
	       internalization.
	    */
	    
	    /* Now we mangle the static keyword .... */
	    kw = __realloc_static_kw(kw , occurence);
	    
	    if (internalize_state) {  
	      stringlist_append_copy( enkf_state->restart_kw_list , kw);
	      
	      if (!ensemble_config_has_key(enkf_state->ensemble_config , kw)) 
		ensemble_config_add_node(enkf_state->ensemble_config , kw , STATIC_STATE , STATIC , NULL , NULL , NULL);
	      
	      if (!enkf_state_has_node(enkf_state , kw)) {
		const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_state->ensemble_config , kw);
		enkf_state_add_node(enkf_state , kw , config_node); 
	      }
	      
	      /* 
		 The following thing can happen:
		 
		 1. A static keyword appears at report step n, and is added to the enkf_state
		    object.
		 
		 2. At report step n+k that static keyword is no longer active, and it is
		    consequently no longer part of restart_kw_list().
		 
		 3. However it is still part of the enkf_state. Not loaded here, and subsequently
	            purged from enkf_main.
	       
		 One keyword where this occurs is FIPOIL, which at least might appear only in the
		 first restart file. Unused static keywords of this type are purged from the
		 enkf_main object by a call to enkf_main_del_unused_static(). The purge is based on
		 looking at the internal __report_step state of the static kw.
	      */
	      
	      {
		enkf_node_type * enkf_node         = enkf_state_get_node(enkf_state , kw);
		enkf_node_ecl_load_static(enkf_node , ecl_kw , report_step , my_config->iens);
		/*
		  Static kewyords go straight out ....
		*/
		enkf_fs_fwrite_node(shared_info->fs , enkf_node , report_step , my_config->iens , FORECAST);
		enkf_node_free_data(enkf_node);
	      }
	    }
	  } 
	} else
	  util_abort("%s: hm - something wrong - can (currently) only load FIELD/STATIC implementations from restart files - aborting \n",__func__);
	free(kw);
      }
      enkf_fs_fwrite_restart_kw_list( shared_info->fs , report_step , my_config->iens , enkf_state->restart_kw_list );
    }
  }
  
  /******************************************************************/
  /** 
      Starting on the enkf_node_ecl_load() function calls. This is where the
      actual loading (apart from static keywords) is done. Observe that this
      loading might involve other load functions than the ones used for
      loading PRESSURE++ from ECLIPSE restart files (e.g. for loading seismic
      results..)
  */
  
  {
    hash_iter_type * iter = hash_iter_alloc(enkf_state->node_hash);
    while ( !hash_iter_is_complete(iter) ) {
      enkf_node_type * enkf_node = hash_iter_get_next_value(iter);
      if (enkf_node_get_var_type(enkf_node) == DYNAMIC_STATE) {
	bool internalize_kw = internalize_state;
	if (!internalize_kw)
	  internalize_kw = enkf_node_internalize(enkf_node , report_step);
	
	if (internalize_kw) {
	  if (enkf_node_has_func(enkf_node , ecl_load_func)) {
	    if (enkf_node_ecl_load(enkf_node , run_info->run_path , NULL , restart_file , report_step , my_config->iens ))
              enkf_fs_fwrite_node(shared_info->fs , enkf_node , report_step , my_config->iens , FORECAST);
            else {
              *loadOK = false;
              log_add_fmt_message(shared_info->logh , 3 , "[%03d:%04d] Failed load data for node:%s.",my_config->iens , report_step , enkf_node_get_key( enkf_node ));
            }
	  }
	}
      }
    }                                                                      
    hash_iter_free(iter);
  }
  
  /*****************************************************************/
  /* Cleaning up */
  if (restart_file != NULL) ecl_file_free( restart_file );
}






/**
   This function loads the results from a forward simulations from report_step1
   to report_step2. The details of what to load are in model_config and the
   spesific nodes for special cases.

   Will mainly be called at the end of the forward model, but can also
   be called manually from external scope.
*/
   

void enkf_state_internalize_results(enkf_state_type * enkf_state , int report_step1 , int report_step2 , bool * loadOK) {
  int report_step;
  {
    const model_config_type * model_config = enkf_state->shared_info->model_config;
    for (report_step = report_step1; report_step <= report_step2; report_step++) {
      if (model_config_load_state( model_config , report_step)) 
	enkf_state_internalize_state(enkf_state , model_config , report_step , loadOK);

      if (model_config_load_results( model_config , report_step)) 
	enkf_state_internalize_dynamic_results(enkf_state , model_config , report_step , loadOK);
      
    }
  }
}


void * enkf_state_internalize_results_mt( void * arg ) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  enkf_state_type * enkf_state = arg_pack_iget_ptr( arg_pack , 0 );
  int step1                    = arg_pack_iget_int( arg_pack , 1 );
  int step2                    = arg_pack_iget_int( arg_pack , 2 );
  bool loadOK                  = true;
  
  enkf_state_internalize_results( enkf_state , step1 , step2 , &loadOK);
  
  return NULL;
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
  const bool fmt_file  		       = ecl_config_get_formatted(enkf_state->ecl_config);
  char * restart_file    	       = ecl_util_alloc_filename(run_info->run_path , my_config->eclbase , ECL_RESTART_FILE , fmt_file , run_info->step1);
  fortio_type * fortio   	       = fortio_fopen(restart_file , "w" , ECL_ENDIAN_FLIP , fmt_file);
  const char * kw;
  int          ikw;

  if (stringlist_get_size(enkf_state->restart_kw_list) == 0)
    enkf_fs_fread_restart_kw_list(shared_info->fs , run_info->step1 , my_config->iens , enkf_state->restart_kw_list);

  for (ikw = 0; ikw < stringlist_get_size(enkf_state->restart_kw_list); ikw++) {
    kw = stringlist_iget( enkf_state->restart_kw_list , ikw);
    /* 
       Observe that here we are *ONLY* iterating over the
       restart_kw_list instance, and *NOT* the enkf_state
       instance. I.e. arbitrary dynamic keys, and no-longer-active
       static kewyords should not show up.

       If the restart kw_list asks for a keyword which we do not have,
       we assume it is a static keyword and add it it to the
       enkf_state instance. 
       
       This is a bit unfortunate, because a bug/problem of some sort,
       might be masked (seemingly solved) by adding a static keyword,
       before things blow up completely at a later instant.
    */  
    if (!ensemble_config_has_key(enkf_state->ensemble_config , kw)) 
      ensemble_config_add_node(enkf_state->ensemble_config , kw , STATIC_STATE , STATIC , NULL , NULL , NULL );
    
    if (!enkf_state_has_node(enkf_state , kw)) {
      const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_state->ensemble_config , kw);
      enkf_state_add_node(enkf_state , kw , config_node); 
    }
	
    {
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw); 
      enkf_var_type var_type = enkf_node_get_var_type(enkf_node); 
      if (var_type == STATIC_STATE) 
	//ecl_static_kw_inc_counter(enkf_node_value_ptr(enkf_node) , false , run_info->step1);
	enkf_fs_fread_node(shared_info->fs , enkf_node , run_info->step1 , my_config->iens , run_info->init_state_dynamic);
      
      if (var_type == DYNAMIC_STATE) {
	/* Pressure and saturations */
	if (enkf_node_get_impl_type(enkf_node) == FIELD)
	  enkf_node_ecl_write(enkf_node , NULL , fortio , run_info->step1);
	else 
	  util_abort("%s: internal error wrong implementetion type:%d - node:%s aborting \n",__func__ , enkf_node_get_impl_type(enkf_node) , enkf_node_get_key(enkf_node));
      } else if (var_type == STATIC_STATE) {
	enkf_node_ecl_write(enkf_node , NULL , fortio , run_info->step1);
	enkf_node_free_data(enkf_node); /* Just immediately discard the static data. */
      } else {
	fprintf(stderr,"var_type: %d \n",var_type);
	fprintf(stderr,"node    : %s \n",enkf_node_get_key(enkf_node));
	util_abort("%s: internal error - should not be here ... \n",__func__);
      }
      
    }
  }
  fortio_fclose(fortio);
  free(restart_file);
}



/**
  This function writes out all the files needed by an ECLIPSE simulation, this
  includes the restart file, and the various INCLUDE files corresponding to
  parameteres estimated by EnKF.

  The writing of restart file is delegated to enkf_state_write_restart_file().
*/

void enkf_state_ecl_write(enkf_state_type * enkf_state) {
  const run_info_type * run_info         = enkf_state->run_info;
  
  if (run_info->step1 > 0)
    enkf_state_write_restart_file(enkf_state);
  else {
    /*
      These keywords are added here becasue otherwise the main loop
      below will try to write them with ecl_write - and that will fail
      (for report_step 0).
    */
    stringlist_append_copy(enkf_state->restart_kw_list , "SWAT");
    stringlist_append_copy(enkf_state->restart_kw_list , "SGAS");
    stringlist_append_copy(enkf_state->restart_kw_list , "PRESSURE");
    stringlist_append_copy(enkf_state->restart_kw_list , "RV");
    stringlist_append_copy(enkf_state->restart_kw_list , "RS");
  }

  util_make_path(run_info->run_path);
  {
    /** 
	This iteration manipulates the hash (thorugh the enkf_state_del_node() call) 
	
	-----------------------------------------------------------------------------------------
	T H I S  W I L L  D E A D L O C K  I F  T H E   H A S H _ I T E R  A P I   I S   U S E D.
	-----------------------------------------------------------------------------------------
    */
    
    const int num_keys = hash_get_size(enkf_state->node_hash);
    char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
    int ikey;
    
    for (ikey = 0; ikey < num_keys; ikey++) {
      if (!stringlist_contains(enkf_state->restart_kw_list , key_list[ikey])) {          /* Make sure that the elements in the restart file are not written (again). */
	enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
	if (enkf_node_get_var_type( enkf_node ) != STATIC_STATE)                          /* Ensure that no-longer-active static keywords do not create problems. */
	  enkf_node_ecl_write(enkf_node , run_info->run_path , NULL , run_info->step1); 
      }
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



/**
   Which state to load on the next step must be much more
   fine-grained. This is a fucking hack.
*/

static void enkf_state_try_fread(enkf_state_type * enkf_state , int mask , int report_step , state_enum load_state) {
  shared_info_type * shared_info = enkf_state->shared_info;
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;
  
  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_include_type(enkf_node , mask)) {
      enkf_var_type var_type = enkf_node_get_var_type( enkf_node );
      if ((var_type == PARAMETER) || (var_type == STATIC_STATE))
	enkf_fs_fread_node(shared_info->fs , enkf_node , report_step , my_config->iens , load_state);
      else {
	if (!enkf_fs_try_fread_node(shared_info->fs , enkf_node , report_step , my_config->iens , BOTH)) {
          //printf("%s: failed to load node:%s  report_step:%d iens:%d \n",__func__ , key_list[ikey] , report_step , enkf_state->my_config->iens  );
	  util_abort("%s: failed to load node:%s  report_step:%d iens:%d \n",__func__ , key_list[ikey] , report_step , enkf_state->my_config->iens  );
        }
      }
    }
  }
  util_free_stringlist(key_list , num_keys);
}




/**
   This is a special function which is only used to load the initial
   state of dynamic_state nodes. It checks if the enkf_config_node has
   set a valid value for input_file, in that case that means we should
   also have an internalized representation of it, otherwise it will
   just return (i.e. for PRESSURE / SWAT).
*/

static void enkf_state_fread_initial_state(enkf_state_type * enkf_state) {
  shared_info_type * shared_info = enkf_state->shared_info;
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;
  
  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_get_var_type(enkf_node) == DYNAMIC_STATE) {
      const enkf_config_node_type * config_node = enkf_node_get_config( enkf_node );

      /* Just checked for != NULL */
      char * load_file = enkf_config_node_alloc_infile( config_node , 0);
      if (load_file != NULL) 
	enkf_fs_fread_node(shared_info->fs , enkf_node , 0 , my_config->iens , ANALYZED);

      util_safe_free( load_file );
    }
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
      enkf_state_del_node(enkf_state , enkf_node_get_key(enkf_node));
  }                                                                     
  util_free_stringlist(key_list , num_keys);
}

      




void enkf_state_free(enkf_state_type *enkf_state) {
  hash_free(enkf_state->node_hash);
  subst_list_free(enkf_state->subst_list);
  hash_free(enkf_state->subst_description);
  stringlist_free(enkf_state->restart_kw_list);
  member_config_free(enkf_state->my_config);
  run_info_free(enkf_state->run_info);
  shared_info_free(enkf_state->shared_info);
  forward_model_free(enkf_state->default_forward_model);
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
    fprintf(stderr,"%s: tried to remove node:%s which is not in state - internal error?? \n",__func__ , node_key);
}


/**
   This function will set all the subst_kw key=value pairs which
   change with report step.
*/

static void enkf_state_set_dynamic_subst_kw(enkf_state_type * enkf_state , const char * run_path , int step1 , int step2) {
  const member_config_type  * my_config = enkf_state->my_config;  
  const bool fmt_file  = ecl_config_get_formatted(enkf_state->ecl_config);
  char * step1_s 	   = util_alloc_sprintf("%d" , step1);
  char * step2_s 	   = util_alloc_sprintf("%d" , step2);
  char * step1_s04 	   = util_alloc_sprintf("%04d" , step1);
  char * step2_s04 	   = util_alloc_sprintf("%04d" , step2);
  char * restart_file1     = ecl_util_alloc_filename(NULL , my_config->eclbase , ECL_RESTART_FILE , fmt_file , step1);
  char * restart_file2     = ecl_util_alloc_filename(NULL , my_config->eclbase , ECL_RESTART_FILE , fmt_file , step2);
  
  enkf_state_add_subst_kw(enkf_state , "RUNPATH"       , run_path      , NULL);
  enkf_state_add_subst_kw(enkf_state , "TSTEP1"        , step1_s       , NULL);
  enkf_state_add_subst_kw(enkf_state , "TSTEP2"        , step2_s       , NULL);
  enkf_state_add_subst_kw(enkf_state , "TSTEP1_04"     , step1_s04     , NULL);
  enkf_state_add_subst_kw(enkf_state , "TSTEP2_04"     , step2_s04     , NULL);
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE1" , restart_file1 , NULL);
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE2" , restart_file2 , NULL);

  /**
     The <INIT> magic string:
  */
  if (step1 == 0) {
    const char * init_file = ecl_config_get_equil_init_file(enkf_state->ecl_config);
    if (init_file == NULL) 
      util_abort("%s: INIT_SECTION is not set - must either use INIT_SECTION in config_file or EQUIL keyword.",__func__);
    
    if (init_file != NULL) {
      char * tmp_include = util_alloc_sprintf("INCLUDE\n   \'%s\' /\n",init_file);
      enkf_state_add_subst_kw(enkf_state , "INIT" , tmp_include , NULL);
      free(tmp_include);
    }
  } else {
    char * data_initialize = util_alloc_sprintf("RESTART\n   \'%s\'  %d  /\n" , my_config->eclbase , step1);
    enkf_state_add_subst_kw(enkf_state , "INIT" , data_initialize , NULL);
    free(data_initialize);
  }
  

  free(step1_s);
  free(step2_s);
  free(step1_s04);
  free(step2_s04);
  free(restart_file1);
  free(restart_file2);
}



void enkf_state_printf_subst_list(enkf_state_type * enkf_state , int step1 , int step2) {
  int ikw;
  const char * fmt_string = "%-16s %-40s :: %s\n";
  printf("\n\n");
  printf(fmt_string , "Key" , "Current value" , "Description");
  printf("------------------------------------------------------------------------------------------------------------------------\n");
  if (step1 >= 0)
    enkf_state_set_dynamic_subst_kw(enkf_state , NULL , step1 , step2);

  for (ikw = 0; ikw < subst_list_get_size( enkf_state->subst_list ); ikw++) {
    const char * key   = subst_list_iget_key( enkf_state->subst_list , ikw);
    const char * value = subst_list_iget_value( enkf_state->subst_list , ikw);
    const char * desc  = hash_get( enkf_state->subst_description , key);
    
    if (value != NULL)
      printf(fmt_string , key , value , desc);
    else
      printf(fmt_string , key , "[Not set]" , desc);
  }
  printf("------------------------------------------------------------------------------------------------------------------------\n");
  
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


static void enkf_state_init_eclipse(enkf_state_type *enkf_state) {
  const member_config_type  * my_config = enkf_state->my_config;  
  {
    const run_info_type       * run_info    = enkf_state->run_info;
    if (!run_info->__ready) 
      util_abort("%s: must initialize run parameters with enkf_state_init_run() first \n",__func__);
    
    util_make_path(run_info->run_path);
    {
      char * schedule_file = util_alloc_filename(run_info->run_path , ecl_config_get_schedule_target(enkf_state->ecl_config) , NULL);
      sched_file_fprintf_i(my_config->sched_file , run_info->step2 , schedule_file);
      free(schedule_file);
    }

    /**
       For reruns of various kinds the parameters and the state are
       generally loaded from different timesteps:
    */

    /* Loading parameter information: loaded from timestep: run_info->init_step_parameters. */
    enkf_state_fread(enkf_state , PARAMETER , run_info->init_step_parameters , run_info->init_state_parameter);
    
    
    /* Loading state information: loaded from timestep: run_info->step1 */
    if (run_info->step1 == 0)
      enkf_state_fread_initial_state(enkf_state); 
    else
      enkf_state_try_fread(enkf_state , DYNAMIC_STATE + STATIC_STATE , run_info->step1 , run_info->init_state_dynamic);
    
    enkf_state_set_dynamic_subst_kw(  enkf_state , run_info->run_path , run_info->step1 , run_info->step2);
    ert_templates_instansiate( enkf_state->shared_info->templates , run_info->run_path , enkf_state->subst_list );
    enkf_state_ecl_write( enkf_state );
    
    {
      char * stdin_file = util_alloc_filename(run_info->run_path , "eclipse"  , "stdin");  /* The name eclipse.stdin must be mathched when the job is dispatched. */
      ecl_util_init_stdin( stdin_file , my_config->eclbase );
      free(stdin_file);
    }
    
    /* Writing the ECLIPSE data file. */
    {
      char * data_file = ecl_util_alloc_filename(run_info->run_path , my_config->eclbase , ECL_DATA_FILE , true , -1);
      subst_list_filter_file(enkf_state->subst_list , ecl_config_get_data_file(enkf_state->ecl_config) , data_file);
    }
    
    /* This is where the job script is created */
    forward_model_python_fprintf( enkf_state->forward_model , run_info->run_path , enkf_state->subst_list);
  }
}






/**
   xx_run_forward_model() has been split in two functions:

   1: enkf_state_start_forward_model()

   2: enkf_state_complete_forward_model()

   Because the first is quite CPU intensive (gunzip), and the number of
   concurrent threads should be limited. For the second there is one
   thread for each ensemble member. This is handled by the calling scope.
*/



static void enkf_state_start_forward_model(enkf_state_type * enkf_state) {
  run_info_type       * run_info    = enkf_state->run_info;
  if (run_info->active) {  /* if the job is not active we just return .*/
    const shared_info_type    * shared_info = enkf_state->shared_info;
    const member_config_type  * my_config   = enkf_state->my_config;
    
    /*
      Prepare the job and submit it to the queue
    */
    enkf_state_init_eclipse(enkf_state);
    job_queue_add_job(shared_info->job_queue , run_info->run_path , my_config->eclbase , my_config->iens , forward_model_get_lsf_request(enkf_state->forward_model));
    run_info->num_internal_submit++;
  }
}


/** 
    This function is called when:

     1. The external queue system has said that everything is OK; BUT
        the ert layer  to load all the data.
    
     2. The external queue system has let the job fail.
    
    If resample_when_fail is true the parameter and state variables
    will be resampled before retrying. If resample_when_fail is not
    equal to true it is actually not much of a point to do this - the
    chance that the job will suddenly get through now is quite small.
*/

static bool enkf_state_internal_retry(enkf_state_type * enkf_state , bool load_failure) {
  const member_config_type  * my_config   = enkf_state->my_config;
  run_info_type             * run_info    = enkf_state->run_info;
  const shared_info_type    * shared_info = enkf_state->shared_info;

  if (load_failure)
    log_add_fmt_message(shared_info->logh , 1 , "[%03d:%04d-%04d] Failed to load all data.",my_config->iens , run_info->step1 , run_info->step2);
  else
    log_add_fmt_message(shared_info->logh , 1 , "[%03d:%04d-%04d] Forward model failed.",my_config->iens, run_info->step1 , run_info->step2);

  if (run_info->num_internal_submit < run_info->max_internal_submit) {
    if (run_info->resample_when_fail) {
      log_add_fmt_message( shared_info->logh , 1 , "[%03d] Resampling and resubmitting realization." ,my_config->iens);
      /* resample_when_fail is set to true - we try to resample before resubmitting. */
      {
        stringlist_type * init_keys = ensemble_config_alloc_keylist_from_var_type( enkf_state->ensemble_config , DYNAMIC_STATE + PARAMETER );
        for (int ikey=0; ikey < stringlist_get_size( init_keys ); ikey++) {
          enkf_node_type * node = enkf_state_get_node( enkf_state , stringlist_iget( init_keys , ikey) );
          enkf_node_initialize( node , my_config->iens );
        }
        stringlist_free( init_keys );
      }
    } else
      log_add_fmt_message( shared_info->logh , 1 , "[%03d:%04d-%04d] Retrying realization from ERT main." ,my_config->iens , run_info->step1 , run_info->step2);
    
    enkf_state_ecl_write( enkf_state );  /* Writing a full new enkf_state instance */
    run_info->num_internal_submit++;
    job_queue_set_external_restart( shared_info->job_queue , my_config->iens );

    return true;
  } else
    return false;
}



/** 
    Observe that if run_info == false, this routine will return with
    job_completeOK == true, that might be a bit misleading.
*/

static void enkf_state_complete_forward_model(enkf_state_type * enkf_state) {
  run_info_type             * run_info    = enkf_state->run_info;
  run_info->runOK = true;
  if (run_info->active) {
    const shared_info_type    * shared_info = enkf_state->shared_info;
    const member_config_type  * my_config   = enkf_state->my_config;
    const int usleep_time                   = 2500000; //100000; /* 1/10 of a second */ 
    job_status_type final_status;
    

    /*****************************************************************/
    /* Start of wait loop to wait for the job to complete.           */
    while (true) {
      bool loadOK  = true;
      final_status = job_queue_export_job_status(shared_info->job_queue , my_config->iens);
  
      if (final_status == job_queue_run_OK) {
        /**
           The queue system has reported that the run is OK, i.e. it
           has completed and produced the targetfile it should. We
           then check in this scope whether the results can be loaded
           back; if that is OK the final status is updated, otherwise: restart.
        */
	log_add_fmt_message( shared_info->logh , 2 , "[%03d:%04d-%04d] Forward model complete - starting to load results." , my_config->iens , run_info->step1, run_info->step2);
        enkf_state_internalize_results(enkf_state , run_info->load_start , run_info->step2 , &loadOK); 
        if (loadOK) {
          final_status = job_queue_all_OK;
          job_queue_set_load_OK( shared_info->job_queue , my_config->iens );
          log_add_fmt_message( shared_info->logh , 2 , "[%03d:%04d-%04d] Results loaded successfully." , my_config->iens , run_info->step1, run_info->step2);
          break; /* All coool */
        } else 
          if (!enkf_state_internal_retry( enkf_state , true)) 
            /* 
               We tell the queue system that the job failed hard; it
               will immediately come back here with status
               job_queue_run_FAIL and then it falls all the way
               through to runOK = false and no more attempts.
            */
            job_queue_set_external_fail( shared_info->job_queue , my_config->iens );
      } else if (final_status == job_queue_run_FAIL) {
        /* 
           The external queue system has said that the job failed - we
           give it another try from this scope?? 
        */
        if (!enkf_state_internal_retry( enkf_state , false)) {
          run_info->runOK = false; /* OK - no more attempts. */
          log_add_fmt_message( shared_info->logh , 1 , "[%03d:%04d-%04d] FAILED COMPLETELY." , my_config->iens , run_info->step1, run_info->step2);
          break;
        }
      } else usleep(usleep_time);
    } 
    /* End of wait loop                                              */
    /*****************************************************************/
    
    
    /* In case the job fails, we leave the run_path directory around for debugging. */
    if (final_status == job_queue_all_OK) {
      bool unlink_runpath;
      if (my_config->keep_runpath == DEFAULT_KEEP) {
	if (run_info->run_mode == ENKF_ASSIMILATION)
	  unlink_runpath = true;   /* For assimilation the default is to unlink. */
	else
	  unlink_runpath = false;  /* For experiments the default is to keep the directories around. */
      } else {
	/* We have explcitly set a value for the keep_runpath variable - with either KEEP_RUNAPTH or DELETE_RUNPATH. */
	if (my_config->keep_runpath == EXPLICIT_KEEP)
	  unlink_runpath = false;
	else if (my_config->keep_runpath == EXPLICIT_DELETE)
	  unlink_runpath = true;
	else {
	  util_abort("%s: internal error \n",__func__);
	  unlink_runpath = false; /* Compiler .. */
	}
      }
      if (unlink_runpath)
	util_unlink_path(run_info->run_path);
    }
    run_info_complete_run(enkf_state->run_info);
  }
  run_info->__ready = false;

  /* 
     If we have used a special forward model for this report step we
     discard that model, and recover the default forward_model.
  */
  if (enkf_state->special_forward_model != NULL) {
    forward_model_free( enkf_state->special_forward_model );
    enkf_state->special_forward_model = NULL;
    enkf_state->forward_model = enkf_state->default_forward_model;
  }
}


/**
   Checks that both the run has completed OK - that also includes the
   loading back of results.
*/
   
bool enkf_state_runOK(const enkf_state_type * state) {
  return state->run_info->runOK;
}



void * enkf_state_complete_forward_model__(void * __enkf_state) {
  enkf_state_type * enkf_state = enkf_state_safe_cast(__enkf_state);
  enkf_state_complete_forward_model(enkf_state);             
  return NULL ;
}


void * enkf_state_start_forward_model__(void * __enkf_state) {
  enkf_state_type * enkf_state = enkf_state_safe_cast(__enkf_state);
  enkf_state_start_forward_model( enkf_state );
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
	int elements_added        = enkf_node_serialize(enkf_node , current_serial_offset , serial_vector , &node_complete);
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


//void enkf_state_matrix_serialize(enkf_state_type * enkf_state , const char * key , const active_list_type * active_list, matrix_type * A , int row_offset , int column) {
//  enkf_node_type * enkf_node = enkf_state_get_node( enkf_state , key );
//  enkf_node_matrix_serialize( enkf_node , active_list , A , row_offset , column );
//}
//
//void enkf_state_matrix_deserialize(enkf_state_type * enkf_state , const char * key , const active_list_type * active_list, const matrix_type * A , int row_offset , int column) {
//  enkf_node_type * enkf_node = enkf_state_get_node( enkf_state , key );
//  enkf_node_matrix_deserialize( enkf_node , active_list , A , row_offset , column );
//}





void enkf_ensemble_update(enkf_state_type ** enkf_ensemble , int ens_size , serial_vector_type * serial_vector , const double * X) {
  const int threads = 1;
  int update_mask   = DYNAMIC_STATE + DYNAMIC_RESULT + PARAMETER;
  thread_pool_type * tp = thread_pool_alloc(0 /* threads */);
  enkf_update_info_type ** info_list     = enkf_ensemble_alloc_update_info(enkf_ensemble , ens_size , update_mask , threads , serial_vector);
  int       iens , ithread;


  bool state_complete = false;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_apply_NEW2(enkf_ensemble[iens] ,  update_mask , clear_serial_state_func , NULL);

  while (!state_complete) {
    for (iens = 0; iens < ens_size; iens++) 
      info_list[0]->member_serial_size[iens] = 0;  /* Writing only on element[0] - because that is the only member with actual storage. */
    
    /* Serialize section */
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



void enkf_state_init_run(enkf_state_type * state , 
                         run_mode_type run_mode  , 
                         bool active                    , 
                         bool resample_when_fail , 
                         int max_internal_submit,
                         int init_step_parameter         , 
                         state_enum init_state_parameter , 
                         state_enum init_state_dynamic   , 
                         int load_start          , 
                         int step1               , 
                         int step2               , 
                         forward_model_type * __forward_model) {

  member_config_type * my_config    = state->my_config;
  shared_info_type   * shared_info  = state->shared_info;

  if (init_step_parameter != 0)
    resample_when_fail = false;
  
  run_info_set( state->run_info , 
                run_mode        , 
                active          , 
                resample_when_fail , 
                max_internal_submit,
                init_step_parameter , 
                init_state_parameter , 
                init_state_dynamic  , 
                load_start , 
                step1 , 
                step2 , 
                __forward_model , 
                my_config->iens ,  
                model_config_get_runpath_fmt( shared_info->model_config ),
                state->subst_list );
  
  if (__forward_model != NULL)
    enkf_state_set_special_forward_model( state , __forward_model);
}



/*****************************************************************/

const sched_file_type * enkf_state_get_sched_file(const enkf_state_type * enkf_state) {
  return member_config_get_sched_file(enkf_state->my_config);
}



