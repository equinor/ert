#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/parser.hpp>
#include <ert/util/util.hpp>
#include <ert/util/vector.hpp>

#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>

/*
   This file implements a 'forward-model' object. I
*/

struct forward_model_struct {
    /** The actual jobs in this forward model. */
    vector_type *jobs;
    /** This is the list of external jobs which have been installed - which we
     * can choose from. */
    const ext_joblist_type *ext_joblist;
};

forward_model_type *forward_model_alloc(const ext_joblist_type *ext_joblist) {
    forward_model_type *forward_model =
        (forward_model_type *)util_malloc(sizeof *forward_model);

    forward_model->jobs = vector_alloc_new();
    forward_model->ext_joblist = ext_joblist;

    return forward_model;
}

/**
   Allocates and returns a stringlist with all the names in the
   current forward_model.
*/
stringlist_type *
forward_model_alloc_joblist(const forward_model_type *forward_model) {
    stringlist_type *names = stringlist_alloc_new();
    int i;
    for (i = 0; i < vector_get_size(forward_model->jobs); i++) {
        const ext_job_type *job =
            (const ext_job_type *)vector_iget_const(forward_model->jobs, i);
        stringlist_append_copy(names, ext_job_get_name(job));
    }

    return names;
}

/**
   This function adds the job named 'job_name' to the forward model. The return
   value is the newly created ext_job instance. This can be used to set private
   arguments for this job.
*/
ext_job_type *forward_model_add_job(forward_model_type *forward_model,
                                    const char *job_name) {
    ext_job_type *new_job =
        ext_joblist_get_job_copy(forward_model->ext_joblist, job_name);
    vector_append_owned_ref(forward_model->jobs, new_job, ext_job_free__);
    return new_job;
}

void forward_model_clear(forward_model_type *forward_model) {
    vector_clear(forward_model->jobs);
}

void forward_model_free(forward_model_type *forward_model) {
    vector_free(forward_model->jobs);
    free(forward_model);
}

ext_job_type *forward_model_iget_job(forward_model_type *forward_model,
                                     int index) {
    return (ext_job_type *)vector_iget(forward_model->jobs, index);
}

int forward_model_get_length(const forward_model_type *forward_model) {
    return vector_get_size(forward_model->jobs);
}
