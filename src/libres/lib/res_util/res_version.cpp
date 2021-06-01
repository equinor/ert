#include <ert/util/util.hpp>

#include <ert/res_util/res_version.hpp>

#define xstr(s) #s
#define str(s) xstr(s)

const char* res_version_get_git_commit() {
    #ifdef GIT_COMMIT
        return str(GIT_COMMIT);
    #else
        return "Unknown git commit hash";
    #endif
}

const char* res_version_get_build_time() {
    #ifdef COMPILE_TIME_STAMP
        return COMPILE_TIME_STAMP;
    #else
        return "Unknown build time";
    #endif
}

int res_version_get_major_version() {
  return RES_VERSION_MAJOR;
}


int res_version_get_minor_version() {
  return RES_VERSION_MINOR;
}


const char * res_version_get_micro_version() {
  return str(RES_VERSION_MICRO);
}


bool res_version_is_devel_version() {
    return util_sscanf_int(str(RES_VERSION_MICRO), NULL);
}
