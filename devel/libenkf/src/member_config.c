#include <stdlib.h>
#include <util.h>
#include <member_config.h>
#include <time_t_vector.h>
#include <stdbool.h>
#include <path_fmt.h>
#include <subst.h>
#include <ecl_config.h>
#include <enkf_fs.h>
#include <enkf_types.h>
#include <ensemble_config.h>
#include <time.h>


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
  int                   last_restart_nr;
  time_t_vector_type  * report_time;         /* This vector contains the (per member) report_step -> simulation_time mapping. [NOT in use yet]. */
};




/******************************************************************/
/** Implementation of the member_config struct. All of this implementation
    is private - however some of it is exported through the enkf_state object,
    and it should be perfectly safe to export more of it.
*/



const char * member_config_set_eclbase(member_config_type * member_config , const ecl_config_type * ecl_config , const subst_list_type * subst_list) {
  util_safe_free( member_config->eclbase );
  {
    char * tmp = path_fmt_alloc_path(ecl_config_get_eclbase_fmt(ecl_config), false , member_config->iens);
    member_config->eclbase = subst_list_alloc_filtered_string( subst_list , tmp );
    free( tmp );
  }
  return member_config->eclbase;
}


int member_config_get_iens( const member_config_type * member_config ) {
  return member_config->iens;
}

int member_config_get_last_restart_nr( const member_config_type * member_config) {
  return member_config->last_restart_nr;
}
				   

void member_config_free(member_config_type * member_config) {
  util_safe_free(member_config->eclbase);
  util_safe_free(member_config->casename );

  if (member_config->private_sched_file)
    sched_file_free( member_config->sched_file );
  time_t_vector_free( member_config->report_time );
  free(member_config);
}



static void member_config_set_keep_runpath(member_config_type * member_config , keep_runpath_type keep_runpath) {
  member_config->keep_runpath   = keep_runpath;
}




const sched_file_type * member_config_get_sched_file( const member_config_type * member_config) {
  return member_config->sched_file;
}

keep_runpath_type member_config_get_keep_runpath(const member_config_type * member_config) {
  return member_config->keep_runpath;
}


void member_config_iset_sim_time( member_config_type * member_config , int report_step , time_t sim_time ) {
  time_t_vector_iset( member_config->report_time , report_step , sim_time );
}

/**
   This function will return the default value (i.e. -1) if the input
   report_step is invalid. The calling scope must check for this.
*/

time_t member_config_iget_sim_time( const member_config_type * member_config , int report_step) {
  return time_t_vector_safe_iget( member_config->report_time , report_step );
}


/**
   Will return -1 if the data are not available. 
*/
static time_t member_config_iget_sim_days( const member_config_type * member_config , int report_step) {
  time_t start_time = time_t_vector_iget( member_config->report_time , 0 );
  time_t sim_time   = time_t_vector_iget( member_config->report_time , report_step );
  if (sim_time >= start_time)
    return 1.0 * (sim_time - start_time) / (3600 * 24);
  else
    return -1;
}



bool member_config_has_report_step( const member_config_type * member_config , int report_step) {
  if (time_t_vector_size( member_config->report_time ) > report_step)
    return false;
  else {
    time_t def_time = time_t_vector_get_default( member_config->report_time );
    if ( time_t_vector_iget( member_config->report_time , report_step ) != def_time )
      return true;
    else
      return false;
  }
}



static void member_config_fread_sim_time( member_config_type * member_config , enkf_fs_type * enkf_fs) {
  FILE * stream = enkf_fs_open_excase_member_file( enkf_fs , "sim_time" , member_config->iens );
  if (stream != NULL) {
    time_t_vector_fread( member_config->report_time , stream );
    fclose( stream );
  }
}

void member_config_fwrite_sim_time( const member_config_type * member_config , enkf_fs_type * enkf_fs ) {
  FILE * stream = enkf_fs_open_case_member_file( enkf_fs , "sim_time" , member_config->iens , "w");
  time_t_vector_fwrite( member_config->report_time , stream );
  fclose( stream );
}


const char * member_config_get_eclbase( const member_config_type * member_config ) {
  return member_config->eclbase;
}

const char * member_config_get_casename( const member_config_type * member_config ) {
  return member_config->casename;
}


member_config_type * member_config_alloc(int iens , 
                                         const char * casename , 
                                         keep_runpath_type            keep_runpath , 
                                         const ecl_config_type      * ecl_config , 
                                         const ensemble_config_type * ensemble_config,
                                         enkf_fs_type               * fs) { 
						
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
    member_config->last_restart_nr  = sched_file_get_num_restart_files( member_config->sched_file ) - 1; /* Fuck me +/- 1 */
    member_config->last_restart_nr += ecl_config_get_prediction_length( ecl_config );
  }
  member_config->report_time = time_t_vector_alloc( 0 , -1 );
  member_config_fread_sim_time( member_config , fs );
  return member_config;
}
