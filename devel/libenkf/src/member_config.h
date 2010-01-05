#ifndef __MEMBER_CONFIG_H__
#define __MEMBER_CONFIG_H__

#ifdef __cplusplus 
extern "C" {
#endif

#include <enkf_fs.h>
#include <enkf_types.h>
#include <ensemble_config.h>
#include <ecl_config.h>
#include <time.h>
#include <sched_file.h> 
#include <enkf_types.h>
#include <stdbool.h>
#include <subst_list.h>

typedef struct member_config_struct member_config_type;

int                     member_config_get_sim_length( const member_config_type * member_config );
keep_runpath_type       member_config_get_keep_runpath(const member_config_type * member_config);
int                     member_config_get_iens( const member_config_type * member_config );
void                    member_config_fwrite_sim_time( const member_config_type * member_config , enkf_fs_type * enkf_fs );
void                    member_config_iset_sim_time( member_config_type * member_config , int report_step , time_t sim_time );
double                  member_config_iget_sim_days( member_config_type * member_config , int report_step, enkf_fs_type * fs);
time_t                  member_config_iget_sim_time( member_config_type * member_config , int report_step, enkf_fs_type * fs);
bool                    member_config_has_report_step( const member_config_type * member_config , int report_step);
const char *            member_config_set_eclbase(member_config_type * member_config , const ecl_config_type * ecl_config , const subst_list_type * subst_list);
int                     member_config_get_last_restart_nr( const member_config_type * member_config);
void                    member_config_free(member_config_type * member_config) ;
const char *            member_config_get_eclbase( const member_config_type * member_config );
const char *            member_config_get_casename( const member_config_type * member_config );
const sched_file_type * member_config_get_sched_file( const member_config_type * member_config);

bool                    member_config_pre_clear_runpath(const member_config_type * member_config);

member_config_type *    member_config_alloc(int iens , 
                                            const char * casename , 
                                            bool                         pre_clear_runpath , 
                                            keep_runpath_type            keep_runpath      , 
                                            const ecl_config_type      * ecl_config        , 
                                            const ensemble_config_type * ensemble_config   ,
                                            enkf_fs_type               * fs);


#ifdef __cplusplus 
}
#endif
#endif
