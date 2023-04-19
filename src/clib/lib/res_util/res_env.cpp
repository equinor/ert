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
std::vector<std::string> res_env_alloc_PATH_list() {
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
   This function will take a string as input, and then replace all if
   $VAR expressions with the corresponding environment variable. If
   the environament variable VAR is not set, the string literal $VAR
   is retained. The return value is a newly allocated string.

   If the input value is NULL - the function will just return NULL;
*/
char *res_env_alloc_envvar(const char *value) {
    if (value == NULL)
        return NULL;
    else {
        // Start by filling up a buffer instance with the current content of @value.
        buffer_type *buffer = buffer_alloc(1024);
        buffer_fwrite_char_ptr(buffer, value);
        buffer_rewind(buffer);

        while (true) {
            if (buffer_strchr(buffer, '$')) {
                const char *data = (const char *)buffer_get_data(buffer);
                // offset points at the first character following the '$'
                int offset = buffer_get_offset(buffer) + 1;
                int var_length = 0;

                /* Find the length of the variable name */
                while (true) {
                    char c;
                    c = data[offset + var_length];
                    if (!(isalnum(c) || c == '_'))
                        // Any character which is NOT in the set [a-Z,0-9_]
                        // marks the end of the variable.
                        break;

                    if (c == '\0') /* The end of the string. */
                        break;

                    var_length += 1;
                }

                {
                    char *var_name = util_alloc_substring_copy(
                        data, offset - 1,
                        var_length + 1); /* Include the leading $ */
                    const char *var_value = getenv(&var_name[1]);

                    if (var_value != NULL)
                        // The actual string replacement:
                        buffer_search_replace(buffer, var_name, var_value);
                    else
                        // The variable is not defined, and we leave the $name.
                        buffer_fseek(buffer, var_length, SEEK_CUR);

                    free(var_name);
                }
            } else
                break; /* No more $ to replace */
        }

        buffer_shrink_to_fit(buffer);
        {
            char *expanded_value = (char *)buffer_get_data(buffer);
            buffer_free_container(buffer);
            return expanded_value;
        }
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
