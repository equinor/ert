#include <ert/job_queue/ext_joblist.hpp>
#include <ert/util/test_util.hpp>
#include <stdlib.h>

void load_job_directory(ext_joblist_type *joblist, const char *path) {
    bool user_mode = false;
    ext_joblist_add_jobs_in_directory(joblist, path, user_mode, true);
    test_assert_true(ext_joblist_has_job(joblist, "SYMLINK"));
}

int main(int argc, char **argv) {
    int status = 0;
    ext_joblist_type *joblist = ext_joblist_alloc();
    load_job_directory(joblist, argv[1]);
    ext_joblist_free(joblist);
    exit(status);
}
