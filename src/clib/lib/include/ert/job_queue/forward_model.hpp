#ifndef ERT_FORWARD_MODEL_H
#define ERT_FORWARD_MODEL_H

#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/stringlist.hpp>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/ext_joblist.hpp>

typedef struct forward_model_struct forward_model_type;

extern "C" stringlist_type *
forward_model_alloc_joblist(const forward_model_type *forward_model);
extern "C" PY_USED void forward_model_clear(forward_model_type *forward_model);
extern "C" forward_model_type *
forward_model_alloc(const ext_joblist_type *ext_joblist);
void forward_model_parse_job_args(forward_model_type *model,
                                  const stringlist_type *list,
                                  const subst_list_type *define_args);
void forward_model_parse_job_deprecated_args(
    forward_model_type *forward_model, const char *input_string,
    const subst_list_type *define_args); //DEPRECATED
extern "C" void forward_model_free(forward_model_type *);
extern "C" ext_job_type *
forward_model_iget_job(forward_model_type *forward_model, int index);
extern "C" int
forward_model_get_length(const forward_model_type *forward_model);

extern "C" ext_job_type *
forward_model_add_job(forward_model_type *forward_model, const char *job_name);

#endif
