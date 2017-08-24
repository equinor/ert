#include <ert/util/util.h>
#include <ert/ert/ert_version.h>

#define xstr(s) #s
#define str(s) xstr(s)

const char* ert_version_get_git_commit() {
    #ifdef GIT_COMMIT
        return str(GIT_COMMIT);
    #else
        return "Unknown git commit hash";
    #endif
}

const char* ert_version_get_git_commit_short() {
    #ifdef GIT_COMMIT_SHORT
        return str(GIT_COMMIT_SHORT);
    #else
        return "Unknown git short commit hash";
    #endif
}


const char* ert_version_get_build_time() {
    #ifdef COMPILE_TIME_STAMP
        return COMPILE_TIME_STAMP;
    #else
        return "Unknown build time";
    #endif
}

int ert_version_get_major_version() {
  return ERT_VERSION_MAJOR;
}


int ert_version_get_minor_version() {
  return ERT_VERSION_MINOR;
}


const char * ert_version_get_micro_version() {
  return str(ERT_VERSION_MICRO);
}


bool ert_version_is_devel_version() {
  return util_sscanf_int(str(ERT_VERSION_MICRO), NULL);
}
