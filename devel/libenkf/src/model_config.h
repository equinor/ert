#ifndef __MODEL_CONFIG_H__
#define __MODEL_CONFIG_H__
#include <time.h>
#include <config.h>
#include <ext_joblist.h>
#include <enkf_sched.h>
#include <history.h>
#include <sched_file.h>

typedef struct model_config_struct model_config_type;

model_config_type * model_config_alloc(const config_type * , const ext_joblist_type * , const sched_file_type *);
void                model_config_free(model_config_type *);
enkf_fs_type      * model_config_get_fs(const model_config_type * );
path_fmt_type     * model_config_get_runpath_fmt(const model_config_type * );
char              * model_config_alloc_result_path(const model_config_type *  , int );
enkf_sched_type   * model_config_get_enkf_sched(const model_config_type * );
history_type      * model_config_get_history(const model_config_type * );
#endif
