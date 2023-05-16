#include <filesystem>
#include <string>
#include <vector>

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include <ert/res_util/res_env.hpp>
#include <ert/res_util/string.hpp>
#include <ert/util/buffer.hpp>
#include <ert/util/util.hpp>

namespace fs = std::filesystem;

void res_env_unsetenv(const char *variable) { unsetenv(variable); }

void res_env_setenv(const char *variable, const char *value) {
    int overwrite = 1;
    setenv(variable, value, overwrite);
}

/**
   Will return a NULL terminated list char ** of the paths in the PATH
   variable.
*/
static std::vector<std::string> res_env_alloc_PATH_list() {
    char *path_env = getenv("PATH");
    if (path_env != NULL) {
        return ert::split(path_env, ':');
    }
    return std::vector<std::string>();
}

/**
   This function searches through the content of the (currently set)
   PATH variable, and allocates a string containing the full path
   (first match) to the executable given as input.

   * If the entered executable already is an absolute path, a copy of
     the input is returned *WITHOUT* consulting the PATH variable (or
     checking that it exists).

   * If the executable starts with "./" getenv("PWD") is prepended.

   * If the executable is not found in the PATH list NULL is returned.
*/
char *res_env_alloc_PATH_executable(const char *executable) {
    if (util_is_abs_path(executable)) {
        if (util_is_executable(executable))
            return util_alloc_string_copy(executable);
        else
            return NULL;
    } else if (strncmp(executable, "./", 2) == 0) {
        char *cwd = util_alloc_cwd();
        char *path = util_alloc_filename(cwd, &executable[2], NULL);

        /* The program has been invoked as ./xxxx */
        if (!(util_is_file(path) && util_is_executable(path))) {
            free(path);
            path = NULL;
        }
        free(cwd);

        return path;
    } else {
        char *full_path = NULL;
        auto path_list = res_env_alloc_PATH_list();
        int ipath = 0;

        for (auto path : path_list) {
            try {
                auto dir_iter = fs::directory_iterator(fs::path(path));
            } catch (fs::filesystem_error &err) {
                continue;
            }
            char *current_attempt =
                util_alloc_filename(path.c_str(), executable, NULL);

            if (util_is_file(current_attempt) &&
                util_is_executable(current_attempt)) {
                full_path = current_attempt;
                break;
            } else {
                free(current_attempt);
                ipath++;
            }
        }

        return full_path;
    }
}

/**
   This function will allocate a string copy of the env_index'th
   occurence of an embedded environment variable from the input
   string.

   An environment variable is defined as follows:

     1. It starts with '$'.
     2. It ends with a characeter NOT in the set [a-Z,0-9,_].

   The function will return environment variable number 'env_index'. If
   no such environment variable can be found in the string the
   function will return NULL.

   Observe that the returned string will start with '$'. This is to
   simplify subsequent calls to util_string_replace_XXX() functions,
   however &ret_value[1] must be used in the subsequent getenv() call:

   {
      char * env_var = res_env_isscanf_alloc_envvar( s , 0 );
      if (env_var != NULL) {
         const char * env_value = getenv( &env_var[1] );   // Skip the leading '$'.
         if (env_value != NULL)
            util_string_replace_inplace( s , env_value );
         else
            fprintf(stderr,"** Warning: environment variable: \'%s\' is not defined \n", env_var);
         free( env_var );
      }
   }


*/
char *res_env_isscanf_alloc_envvar(const char *string, int env_index) {
    int env_count = 0;
    const char *offset = string;
    const char *env_ptr;
    do {
        env_ptr = strchr(offset, '$');
        offset = &env_ptr[1];
        env_count++;
    } while ((env_count <= env_index) && (env_ptr != NULL));

    if (env_ptr != NULL) {
        // We found an environment variable we are interested in. Find the
        // end of this variable and return a copy.
        int length = 1;
        bool cont = true;
        do {

            if (!(isalnum(env_ptr[length]) || env_ptr[length] == '_'))
                cont = false;
            else
                length++;
            if (length == strlen(env_ptr))
                cont = false;
        } while (cont);

        return util_alloc_substring_copy(env_ptr, 0, length);
    } else
        return NULL; /* Could not find any env variable occurences. */
}
