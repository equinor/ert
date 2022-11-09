#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/util/hash.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/ext_joblist.hpp>

//#define MODULE_NAME    "jobs.py"
//#define JOBLIST_NAME   "jobList"

struct ext_joblist_struct {
    hash_type *jobs;
};

ext_joblist_type *ext_joblist_alloc() {
    ext_joblist_type *joblist =
        (ext_joblist_type *)util_malloc(sizeof *joblist);
    joblist->jobs = hash_alloc();
    return joblist;
}

void ext_joblist_free(ext_joblist_type *joblist) {
    hash_free(joblist->jobs);
    free(joblist);
}

void ext_joblist_add_job(ext_joblist_type *joblist, const char *name,
                         ext_job_type *new_job) {
    hash_insert_hash_owned_ref(joblist->jobs, name, new_job, ext_job_free__);
}

ext_job_type *ext_joblist_get_job(const ext_joblist_type *joblist,
                                  const char *job_name) {
    if (hash_has_key(joblist->jobs, job_name))
        return (ext_job_type *)hash_get(joblist->jobs, job_name);
    else {
        util_abort("%s: asked for job:%s which does not exist\n", __func__,
                   job_name);
        return NULL;
    }
}

ext_job_type *ext_joblist_get_job_copy(const ext_joblist_type *joblist,
                                       const char *job_name) {
    if (hash_has_key(joblist->jobs, job_name))
        return ext_job_alloc_copy(
            (const ext_job_type *)hash_get(joblist->jobs, job_name));
    else {
        util_abort("%s: asked for job:%s which does not exist\n", __func__,
                   job_name);
        return NULL;
    }
}

bool ext_joblist_has_job(const ext_joblist_type *joblist,
                         const char *job_name) {
    return hash_has_key(joblist->jobs, job_name);
}

stringlist_type *ext_joblist_alloc_list(const ext_joblist_type *joblist) {
    return hash_alloc_stringlist(joblist->jobs);
}

/**
   Will attempt to remove the job @job_name from the joblist; if the
   job is marked as a shared_job (i.e. installed centrally) the user
   is not allowed to delete it. In this case the function will fail
   silently.

   Returns true if the job is actually removed, and false otherwise.
*/
bool ext_joblist_del_job(ext_joblist_type *joblist, const char *job_name) {
    ext_job_type *job = ext_joblist_get_job(joblist, job_name);
    if (!ext_job_is_shared(job)) {
        hash_del(joblist->jobs, job_name);
        return true;
    } else
        return false;
}

void ext_joblist_add_jobs_in_directory(ext_joblist_type *joblist,
                                       const char *path, bool user_mode,
                                       bool search_path) {
    DIR *dirH = opendir(path);
    if (dirH) {
        while (true) {
            struct dirent *entry = readdir(dirH);
            if (entry != NULL) {
                if ((strcmp(entry->d_name, ".") != 0) &&
                    (strcmp(entry->d_name, "..") != 0)) {
                    char *full_path =
                        (char *)util_alloc_filename(path, entry->d_name, NULL);
                    if (util_is_file(full_path)) {
                        ext_job_type *new_job = ext_job_fscanf_alloc(
                            entry->d_name, user_mode, full_path, search_path);
                        if (new_job != NULL) {
                            ext_joblist_add_job(joblist, entry->d_name,
                                                new_job);
                        } else {
                            fprintf(stderr,
                                    " Failed to add forward model job: %s \n",
                                    full_path);
                        }
                    }
                    free(full_path);
                }
            } else
                break;
        }
        closedir(dirH);
    } else
        fprintf(stderr, "** Warning: failed to open jobs directory: %s\n",
                path);
}

int ext_joblist_get_size(const ext_joblist_type *joblist) {
    return hash_get_size(joblist->jobs);
}
