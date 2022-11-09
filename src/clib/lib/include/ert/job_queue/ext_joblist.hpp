#ifndef ERT_EXT_JOBLIST_H
#define ERT_EXT_JOBLIST_H
#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>

#include <ert/job_queue/ext_job.hpp>

typedef struct ext_joblist_struct ext_joblist_type;

extern "C" ext_joblist_type *ext_joblist_alloc();
extern "C" void ext_joblist_free(ext_joblist_type *);
extern "C" void ext_joblist_add_job(ext_joblist_type *joblist, const char *name,
                                    ext_job_type *new_job);
extern "C" ext_job_type *ext_joblist_get_job(const ext_joblist_type *,
                                             const char *);
extern "C" ext_job_type *ext_joblist_get_job_copy(const ext_joblist_type *,
                                                  const char *);
extern "C" bool ext_joblist_has_job(const ext_joblist_type *, const char *);
extern "C" stringlist_type *
ext_joblist_alloc_list(const ext_joblist_type *joblist);
extern "C" bool ext_joblist_del_job(ext_joblist_type *joblist,
                                    const char *job_name);
void ext_joblist_add_jobs_in_directory(ext_joblist_type *joblist,
                                       const char *path, bool user_mode,
                                       bool search_path);
extern "C" int ext_joblist_get_size(const ext_joblist_type *joblist);
#endif
