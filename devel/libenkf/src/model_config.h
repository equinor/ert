#ifndef __MODEL_CONFIG_H__
#define __MODEL_CONFIG_H__
#include <time.h>
#include <config.h>
#include <ext_joblist.h>
#include <enkf_sched.h>
#include <history.h>
#include <sched_file.h>
#include <path_fmt.h>
#include <forward_model.h>

typedef struct model_config_struct model_config_type;

void                  model_config_set_plot_path(model_config_type * , const char *);
const char          * model_config_get_plot_path(const model_config_type * );
void                  enkf_fs_fwrite_new_mount_map(const char * , const char * );
model_config_type   * model_config_alloc(const config_type * , const ext_joblist_type * , const sched_file_type * , bool);
void                  model_config_free(model_config_type *);
enkf_fs_type        * model_config_get_fs(const model_config_type * );
path_fmt_type       * model_config_get_runpath_fmt(const model_config_type * );
char                * model_config_alloc_result_path(const model_config_type *  , int );
enkf_sched_type     * model_config_get_enkf_sched(const model_config_type * );
history_type        * model_config_get_history(const model_config_type * );
void                  model_config_interactive_set_runpath__(void * arg);
forward_model_type  * model_config_get_std_forward_model( const model_config_type * );
#endif
