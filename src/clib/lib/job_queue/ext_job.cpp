#include <filesystem>

#include <ert/res_util/file_utils.hpp>
#include <ert/res_util/res_env.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/util.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <ert/config/config_parser.hpp>
#include <ert/except.hpp>

#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/job_kw_definitions.hpp>

namespace fs = std::filesystem;

/*
  About arguments
  ---------------
  How a job is run is defined in terms of the following variables:

   o stdout_file / stdin_file / stderr_file
   o arglist
   o ....

  These variables will then contain string values from when the job
  configuration is read in, for example this little job

    STDOUT   my_stdout
    STDERR   my_stderr
    ARGLIST  my_job_input   my_job_output

  stdout & stderr are redirected to the files 'my_stdout' and
  'my_stderr' respectively, and when invoked with an exec() call the
  job is given the argumentlist:

       my_job_input   my_job_output

  This implies that _every_time_ this job is invoked the argumentlist
  will be identical; that is clearly quite limiting! To solve this we
  have the possibility of performing string substitutions on the
  strings in the job defintion prior to executing the job, this is
  handled with the privat_args substitutions. The definition for a
  copy-file job:


    EXECUTABLE   /bin/cp
    ARGLIST      <SRC_FILE>  <TARGET_FILE>


  This can then be invoked several times, with different key=value
  arguments for the SRC_FILE and TARGET_FILE:


      COPY_FILE(SRC_FILE = file1 , TARGET_FILE = /tmp/file1)
      COPY_FILE(SRC_FILE = file2 , TARGET_FILE = /tmp/file2)

*/
/*
  More on STDOUT/STDERR
  ---------------------
  If STDOUT/STDERR is is not defined, output is directed to:
  JOB_NAME.stdout.x or JOB_NAME.stderr.x, respectively

  STDOUT null   directs output to screen
  STDERR null   directs error messages to screen
*/
/*


jobList = [
    {"executable"  : None,
     "environment" : {"LM_LICENSE_PATH" : "user@some.example.com:user@other.example.com",
                      "F_UFMTENDIAN"    : "big"},
     "target_file":"222",
     "argList"   : [],
     "stdout"    : "eclipse.stdout",
     "stderr"    : "eclipse.stdout",
     "stdin"     : "eclipse.stdin"}]
*/

struct ext_job_struct {
    char *name;
    char *executable;
    char *target_file;
    /** Job has failed if this is present. */
    char *error_file;
    /** Will not start if not this file is present */
    char *start_file;
    char *stdout_file;
    char *stdin_file;
    char *stderr_file;
    /** If this is NULL - it will be unrestricted ... */
    char *license_path;
    char *license_root_path;
    char *config_file;
    /** 0 means unlimited. */
    int max_running;
    /** The maximum number of minutes this job is allowed to run - 0:
     * unlimited. */
    int max_running_minutes;

    int min_arg;
    int max_arg;
    int_vector_type *arg_types;
    /** Currently not in use, but will replace deprected_argv */
    stringlist_type *argv;
    /** A substitution list of input arguments which is performed before the
     * external substitutions - these are the arguments supplied as key=value
     * pairs in the forward model call. */
    subst_list_type *private_args;
    const subst_list_type *define_args;
    char *private_args_string;
    char *argv_string;
    /** This should *NOT* start with the executable */
    stringlist_type *deprecated_argv;
    hash_type *environment;
    hash_type *default_mapping;
    hash_type *exec_env;
    char *help_text;

    /** Can the current user/delete this job? (private_job == true) means the
     * user can edit it. */
    bool private_job;
    /** Temporary variable consulted during the bootstrap - when the ext_job is
     * completely initialized this should NOT be consulted anymore. */
    bool __valid;
};

static ext_job_type *ext_job_alloc__(const char *name,
                                     const char *license_root_path,
                                     bool private_job) {
    ext_job_type *ext_job = (ext_job_type *)util_malloc(sizeof *ext_job);

    ext_job->name = util_alloc_string_copy(name);
    ext_job->license_root_path = util_alloc_string_copy(license_root_path);
    ext_job->executable = NULL;
    ext_job->stdout_file = NULL;
    ext_job->target_file = NULL;
    ext_job->error_file = NULL;
    ext_job->start_file = NULL;
    ext_job->stdin_file = NULL;
    ext_job->stderr_file = NULL;
    ext_job->environment = hash_alloc();
    ext_job->default_mapping = hash_alloc();
    ext_job->exec_env = hash_alloc();
    ext_job->argv = stringlist_alloc_new();
    ext_job->define_args = NULL;
    ext_job->deprecated_argv = NULL;
    ext_job->argv_string = NULL;
    ext_job->__valid = true;
    ext_job->license_path = NULL;
    ext_job->config_file = NULL;
    ext_job->max_running = 0;         /* 0 means unlimited. */
    ext_job->max_running_minutes = 0; /* 0 means unlimited. */
    ext_job->min_arg = -1;
    ext_job->max_arg = -1;
    ext_job->arg_types = int_vector_alloc(0, CONFIG_STRING);
    ext_job->private_job =
        private_job; /* If private_job == true the job is user editable. */
    ext_job->help_text = NULL;
    ext_job->private_args_string = NULL;

    // ext_job->private_args is set explicitly in the ext_job_alloc()
    // and ext_job_alloc_copy() functions.
    return ext_job;
}

void ext_job_free_deprecated_argv(ext_job_type *ext_job) {
    if (ext_job->deprecated_argv) {
        stringlist_free(ext_job->deprecated_argv);
        ext_job->deprecated_argv = NULL;
    }
}

const char *ext_job_get_help_text(const ext_job_type *job) {
    if (job->help_text != NULL)
        return job->help_text;
    else
        return "No help text installed for this job.";
}

void ext_job_set_help_text(ext_job_type *job, const char *help_text) {
    job->help_text = util_realloc_string_copy(job->help_text, help_text);
}

/*
   Exported function - must have name != NULL. Observe that the
   instance returned from this function is not really usable for
   anything.

   Should probably define a minium set of parameters which must be set
   before the job is in a valid initialized state.
*/

ext_job_type *ext_job_alloc(const char *name, const char *license_root_path,
                            bool private_job) {
    ext_job_type *ext_job =
        ext_job_alloc__(name, license_root_path, private_job);
    ext_job->private_args = subst_list_alloc(NULL);
    return ext_job;
}

ext_job_type *ext_job_alloc_copy(const ext_job_type *src_job) {
    ext_job_type *new_job =
        ext_job_alloc__(src_job->name, src_job->license_root_path,
                        true /* All copies are by default private jobs. */);

    new_job->config_file = util_alloc_string_copy(src_job->config_file);
    new_job->executable = util_alloc_string_copy(src_job->executable);
    new_job->target_file = util_alloc_string_copy(src_job->target_file);
    new_job->error_file = util_alloc_string_copy(src_job->error_file);
    new_job->start_file = util_alloc_string_copy(src_job->start_file);
    new_job->stdout_file = util_alloc_string_copy(src_job->stdout_file);
    new_job->stdin_file = util_alloc_string_copy(src_job->stdin_file);
    new_job->stderr_file = util_alloc_string_copy(src_job->stderr_file);
    new_job->license_path = util_alloc_string_copy(src_job->license_path);

    ext_job_set_help_text(new_job, src_job->help_text);

    new_job->max_running_minutes = src_job->max_running_minutes;
    new_job->max_running = src_job->max_running;
    new_job->min_arg = src_job->min_arg;
    new_job->max_arg = src_job->max_arg;
    new_job->arg_types = int_vector_alloc_copy(src_job->arg_types);

    new_job->private_args = subst_list_alloc_deep_copy(src_job->private_args);

    /* Copying over all the keys in the environment hash table */
    {
        hash_iter_type *iter = hash_iter_alloc(src_job->environment);
        const char *key = hash_iter_get_next_key(iter);
        while (key != NULL) {
            char *value = (char *)hash_get(src_job->environment, key);
            if (value)
                hash_insert_hash_owned_ref(new_job->environment, key,
                                           util_alloc_string_copy(value), free);
            else
                hash_insert_ref(new_job->environment, key, NULL);
            key = hash_iter_get_next_key(iter);
        }
        hash_iter_free(iter);
    }

    {
        hash_iter_type *iter = hash_iter_alloc(src_job->exec_env);
        const char *key = hash_iter_get_next_key(iter);
        while (key != NULL) {
            char *value = (char *)hash_get(src_job->exec_env, key);
            if (value)
                hash_insert_hash_owned_ref(new_job->exec_env, key,
                                           util_alloc_string_copy(value), free);
            else
                hash_insert_ref(new_job->exec_env, key, NULL);
            key = hash_iter_get_next_key(iter);
        }
        hash_iter_free(iter);
    }

    /* The default mapping. */
    {
        hash_iter_type *iter = hash_iter_alloc(src_job->default_mapping);
        const char *key = hash_iter_get_next_key(iter);
        while (key != NULL) {
            char *value = (char *)hash_get(src_job->default_mapping, key);
            hash_insert_hash_owned_ref(new_job->default_mapping, key,
                                       util_alloc_string_copy(value), free);
            key = hash_iter_get_next_key(iter);
        }
        hash_iter_free(iter);
    }

    if (src_job->deprecated_argv) {
        new_job->deprecated_argv = stringlist_alloc_new();
        stringlist_deep_copy(new_job->deprecated_argv,
                             src_job->deprecated_argv);
    }

    return new_job;
}

void ext_job_free(ext_job_type *ext_job) {
    free(ext_job->name);
    free(ext_job->executable);
    free(ext_job->stdout_file);
    free(ext_job->stdin_file);
    free(ext_job->target_file);
    free(ext_job->error_file);
    free(ext_job->stderr_file);
    free(ext_job->license_path);
    free(ext_job->license_root_path);
    free(ext_job->config_file);
    free(ext_job->argv_string);
    free(ext_job->help_text);
    free(ext_job->private_args_string);

    hash_free(ext_job->default_mapping);
    hash_free(ext_job->environment);
    hash_free(ext_job->exec_env);

    stringlist_free(ext_job->argv);
    if (ext_job->deprecated_argv)
        stringlist_free(ext_job->deprecated_argv);
    subst_list_free(ext_job->private_args);

    int_vector_free(ext_job->arg_types);
    free(ext_job);
}

void ext_job_free__(void *__ext_job) {
    ext_job_free(static_cast<ext_job_type *>(__ext_job));
}

/*
   The license_path =

   root_license_path / job_name / job_name

*/

static void ext_job_init_license_control(ext_job_type *ext_job) {
    if (ext_job->license_path == NULL) {
        ext_job->license_path =
            util_alloc_sprintf("%s%c%s", ext_job->license_root_path,
                               UTIL_PATH_SEP_CHAR, ext_job->name);
        util_make_path(ext_job->license_path);
    }
}

void ext_job_set_max_time(ext_job_type *ext_job, int max_time) {
    ext_job->max_running_minutes = max_time;
}

/**
  This method have the following logic:

     executable_abs exists:
         We store the full path as the executable field of the job; and
         try to update the mode of the full_path executable to make sure it
         is executable.

     executable_abs does not exist, but executable_input exists:
        We have found an executable relative to the current working
        directory. This is deprecated behaviour, support will later be
        removed. Suggest new path to executable to user, relative to job
        description file and do a recursive call to this method, using
        the absolute path as @executable parameter

     executable_abs does not exist, executable_input does not exist and
     is an absolute path:
        Write error message

     executable_abs does not exist, executable_input does not exist and
     is a relative path:
        Search trough the PATH variable to try to locate the executable.
        If found, do a recursive call to this method, using the absolute path
        as executable_abs parameter

  @param executable_abs is either
    - an absolute path read directly from config
    - an absolute path constructed from the relative path from config
      with the assumption that the path was a relative path from the
      location of the job description file to the executable.

  @param executable_input is the raw executable as read from config, unprocessed.
*/
void ext_job_set_executable(ext_job_type *ext_job, const char *executable_abs,
                            const char *executable_input, bool search_path) {

    if (fs::exists(executable_abs)) {
        // The executable_abs parameter points to an existing file; we store
        // the full path as the executable field of the job.
        char *full_path = (char *)util_alloc_realpath(executable_abs);
        ext_job->executable =
            util_realloc_string_copy(ext_job->executable, full_path);
        free(full_path);
    } else if (util_is_abs_path(executable_input)) {
        // If you have given an absolute path (i.e. starting with '/' to a non
        // existing job we mark it as invalid - no possibility to provide
        // context replacement afterwards. The job will be discarded by the
        // calling scope.
        throw exc::invalid_argument("** The executable {} was not found",
                                    executable_input);
    } else {
        if (search_path) {
            /* Go through the PATH variable to try to locate the executable. */
            char *path_executable =
                res_env_alloc_PATH_executable(executable_input);

            if (path_executable != NULL) {
                ext_job_set_executable(ext_job, path_executable, NULL,
                                       search_path);
                free(path_executable);
            } else {
                throw exc::invalid_argument(
                    "** The executable {} was not found", executable_input);
            }
        } else {
            ext_job->executable =
                util_realloc_string_copy(ext_job->executable, executable_input);
        }
    }

    // If in the end we do not have execute rights to the executable:
    // discard the job.
    if (ext_job->executable != NULL) {
        if (fs::exists(executable_abs)) {
            if (!util_is_executable(ext_job->executable)) {
                throw exc::invalid_argument(
                    "** You do not have execute rights to: {}",
                    ext_job->executable);
            }
        } else {
            throw exc::invalid_argument("** The executable {} was not found",
                                        ext_job->executable);
        }
    }
}

/**
   Observe that this does NOT reread the ext_job instance from the new
   config_file.
*/
void ext_job_set_args(ext_job_type *ext_job, const stringlist_type *argv) {
    stringlist_deep_copy(ext_job->argv, argv);
}

void ext_job_set_config_file(ext_job_type *ext_job, const char *config_file) {
    ext_job->config_file =
        util_realloc_string_copy(ext_job->config_file, config_file);
}

const char *ext_job_get_config_file(const ext_job_type *ext_job) {
    return ext_job->config_file;
}

void ext_job_set_target_file(ext_job_type *ext_job, const char *target_file) {
    ext_job->target_file =
        util_realloc_string_copy(ext_job->target_file, target_file);
}

const char *ext_job_get_target_file(const ext_job_type *ext_job) {
    return ext_job->target_file;
}

void ext_job_set_error_file(ext_job_type *ext_job, const char *error_file) {
    ext_job->error_file =
        util_realloc_string_copy(ext_job->error_file, error_file);
}

const char *ext_job_get_error_file(const ext_job_type *ext_job) {
    return ext_job->error_file;
}

const char *ext_job_get_executable(const ext_job_type *ext_job) {
    return ext_job->executable;
}

void ext_job_set_start_file(ext_job_type *ext_job, const char *start_file) {
    ext_job->start_file =
        util_realloc_string_copy(ext_job->start_file, start_file);
}

const char *ext_job_get_start_file(const ext_job_type *ext_job) {
    return ext_job->start_file;
}

const char *ext_job_get_license_path(const ext_job_type *ext_job) {
    return ext_job->license_path;
}

const char *ext_job_get_name(const ext_job_type *ext_job) {
    return ext_job->name;
}
void ext_job_set_stdin_file(ext_job_type *ext_job, const char *stdin_file) {
    ext_job->stdin_file =
        util_realloc_string_copy(ext_job->stdin_file, stdin_file);
}

const char *ext_job_get_stdin_file(const ext_job_type *ext_job) {
    return ext_job->stdin_file;
}

void ext_job_set_stdout_file(ext_job_type *ext_job, const char *stdout_file) {
    if (!util_string_equal(stdout_file, "null"))
        ext_job->stdout_file =
            util_realloc_string_copy(ext_job->stdout_file, stdout_file);
}

const char *ext_job_get_stdout_file(const ext_job_type *ext_job) {
    return ext_job->stdout_file;
}

void ext_job_set_stderr_file(ext_job_type *ext_job, const char *stderr_file) {
    if (strcmp(stderr_file, "null") != 0)
        ext_job->stderr_file =
            util_realloc_string_copy(ext_job->stderr_file, stderr_file);
}

const char *ext_job_get_stderr_file(const ext_job_type *ext_job) {
    return ext_job->stderr_file;
}

void ext_job_set_max_running(ext_job_type *ext_job, int max_running) {
    ext_job->max_running = max_running;
    if (max_running > 0)
        ext_job_init_license_control(ext_job);
}

int ext_job_get_max_running(const ext_job_type *ext_job) {
    return ext_job->max_running;
}

void ext_job_set_max_running_minutes(ext_job_type *ext_job,
                                     int max_running_minutes) {
    ext_job->max_running_minutes = max_running_minutes;
}

int ext_job_get_max_running_minutes(const ext_job_type *ext_job) {
    return ext_job->max_running_minutes;
}

static void ext_job_set_min_arg(ext_job_type *ext_job, int min_arg) {
    ext_job->min_arg = min_arg;
}

static void ext_job_set_max_arg(ext_job_type *ext_job, int max_arg) {
    ext_job->max_arg = max_arg;
}

int ext_job_get_min_arg(const ext_job_type *ext_job) {
    return ext_job->min_arg;
}

int ext_job_get_max_arg(const ext_job_type *ext_job) {
    return ext_job->max_arg;
}

subst_list_type *ext_job_get_private_args(ext_job_type *ext_job) {
    return ext_job->private_args;
}

void ext_job_set_private_arg(ext_job_type *ext_job, const char *key,
                             const char *value) {
    subst_list_append_copy(ext_job->private_args, key, value, NULL);
}

void ext_job_set_define_args(ext_job_type *ext_job,
                             const subst_list_type *define_args) {
    ext_job->define_args = subst_list_alloc_deep_copy(define_args);
}

void ext_job_add_environment(ext_job_type *ext_job, const char *key,
                             const char *value) {
    hash_insert_hash_owned_ref(ext_job->environment, key,
                               util_alloc_string_copy(value), free);
}

void ext_job_clear_environment(ext_job_type *ext_job) {
    hash_clear(ext_job->environment);
}

hash_type *ext_job_get_environment(ext_job_type *ext_job) {
    return ext_job->environment;
}

hash_type *ext_job_get_default_mapping(ext_job_type *ext_job) {
    return ext_job->default_mapping;
}

hash_type *ext_job_get_exec_env(ext_job_type *ext_job) {
    return ext_job->exec_env;
}

static char *__alloc_filtered_string(const char *src_string,
                                     const subst_list_type *private_args,
                                     const subst_list_type *global_args) {
    char *tmp1 = subst_list_alloc_filtered_string(
        private_args, src_string); /* internal filtering first */
    char *tmp2;

    if (global_args != NULL) {
        tmp2 = subst_list_alloc_filtered_string(global_args,
                                                tmp1); /* Global filtering. */
        free(tmp1);
    } else
        tmp2 = tmp1;

    return tmp2;
}

#define PRINT_KEY_STRING(stream, key, value)                                   \
    if (value != NULL) {                                                       \
        fprintf(stream, "%16s ", key);                                         \
        fprintf(stream, "%s\n", value);                                        \
    }

#define PRINT_KEY_INT(stream, key, value)                                      \
    if (value != 0) {                                                          \
        fprintf(stream, "%16s ", key);                                         \
        fprintf(stream, "%d\n", value);                                        \
    }

/**
   Observe that the job will save itself to the internalized
   config_file; if you wish to save to some other place you must call
   ext_job_set_config_file() first.
*/
void ext_job_save(const ext_job_type *ext_job) {
    auto stream = mkdir_fopen(fs::path(ext_job->config_file), "w");

    PRINT_KEY_STRING(stream, "EXECUTABLE", ext_job->executable);
    PRINT_KEY_STRING(stream, "STDIN", ext_job->stdin_file);
    PRINT_KEY_STRING(stream, "STDERR", ext_job->stderr_file);
    PRINT_KEY_STRING(stream, "STDOUT", ext_job->stdout_file);
    PRINT_KEY_STRING(stream, "TARGET_FILE", ext_job->target_file);
    PRINT_KEY_STRING(stream, "START_FILE", ext_job->start_file);
    PRINT_KEY_STRING(stream, "ERROR_FILE", ext_job->error_file);
    PRINT_KEY_INT(stream, "MAX_RUNNING", ext_job->max_running);
    PRINT_KEY_INT(stream, "MAX_RUNNING_MINUTES", ext_job->max_running_minutes);

    stringlist_type *list;
    if (ext_job->deprecated_argv)
        list = ext_job->deprecated_argv;
    else
        list = ext_job->argv;

    if (stringlist_get_size(list) > 0) {
        fprintf(stream, "%16s", "ARGLIST");
        stringlist_fprintf(list, " ", stream);
        fprintf(stream, "\n");
    }
    if (hash_get_size(ext_job->environment) > 0) {
        hash_iter_type *hash_iter = hash_iter_alloc(ext_job->environment);
        while (!hash_iter_is_complete(hash_iter)) {
            const char *key = hash_iter_get_next_key(hash_iter);
            fprintf(stream, "%16s  %16s  %s\n", "ENV", key,
                    (const char *)hash_get(ext_job->environment, key));
        }
        hash_iter_free(hash_iter);
    }
    fclose(stream);
}

#undef PRINT_KEY_STRING
#undef PRINT_KEY_INT

void ext_job_fprintf(const ext_job_type *ext_job, FILE *stream) {
    fprintf(stream, "%s", ext_job->name);
    if (subst_list_get_size(ext_job->private_args) > 0) {
        fprintf(stream, "(");
        subst_list_fprintf(ext_job->private_args, stream);
        fprintf(stream, ")");
    }
    fprintf(stream, "  ");
}

config_item_types ext_job_iget_argtype(const ext_job_type *ext_job, int index) {
    return (config_item_types)int_vector_safe_iget(ext_job->arg_types, index);
}

static void ext_job_iset_argtype_string(ext_job_type *ext_job, int iarg,
                                        const char *arg_type) {
    config_item_types type = job_kw_get_type(arg_type);
    if (type != CONFIG_INVALID)
        int_vector_iset(ext_job->arg_types, iarg, type);
}

ext_job_type *ext_job_fscanf_alloc(const char *name,
                                   const char *license_root_path,
                                   bool private_job, const char *config_file,
                                   bool search_path) {

    if (util_entry_readable(config_file)) {
        ext_job_type *ext_job = NULL;
        config_parser_type *config = config_alloc();

        {
            config_schema_item_type *item;
            item = config_add_schema_item(config, "MAX_RUNNING", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_INT);
            item = config_add_schema_item(config, "STDIN", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, "STDOUT", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, "STDERR", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, EXECUTABLE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_PATH);
            item = config_add_schema_item(config, "PORTABLE_EXE", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_PATH);
            item = config_add_schema_item(config, "TARGET_FILE", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, "ERROR_FILE", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, "START_FILE", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            item = config_add_schema_item(config, "ENV", false);
            config_schema_item_set_argc_minmax(item, 1, 2);
            item = config_add_schema_item(config, "EXEC_ENV", false);
            config_schema_item_set_argc_minmax(item, 1, 2);
            item = config_add_schema_item(config, "DEFAULT", false);
            config_schema_item_set_argc_minmax(item, 2, 2);
            item = config_add_schema_item(config, "ARGLIST", false);
            config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
            item = config_add_schema_item(config, "MAX_RUNNING_MINUTES", false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_INT);
            item = config_add_schema_item(config, MIN_ARG_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_INT);
            item = config_add_schema_item(config, MAX_ARG_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_INT);
            item = config_add_schema_item(config, ARG_TYPE_KEY, false);
            config_schema_item_set_argc_minmax(item, 2, 2);
            config_schema_item_iset_type(item, 0, CONFIG_INT);

            stringlist_type *var_types = stringlist_alloc_new();
            stringlist_append_copy(var_types, JOB_STRING_TYPE);
            stringlist_append_copy(var_types, JOB_INT_TYPE);
            stringlist_append_copy(var_types, JOB_FLOAT_TYPE);
            stringlist_append_copy(var_types, JOB_BOOL_TYPE);
            stringlist_append_copy(var_types, JOB_RUNTIME_FILE_TYPE);
            stringlist_append_copy(var_types, JOB_RUNTIME_INT_TYPE);

            config_schema_item_set_indexed_selection_set(item, 1, var_types);
            stringlist_free(var_types);
            config_parser_deprecate(
                config, "PORTABLE_EXE",
                "'PORTABLE_EXE' is deprecated. Use 'EXECUTABLE' instead.");
        }
        {
            config_content_type *content =
                config_parse(config, config_file, "--", NULL, NULL, NULL,
                             CONFIG_UNRECOGNIZED_WARN, true);
            if (config_content_is_valid(content)) {
                ext_job = ext_job_alloc(name, license_root_path, private_job);
                ext_job_set_config_file(ext_job, config_file);

                const stringlist_type *warnings =
                    config_content_get_warnings(content);
                if (stringlist_get_size(warnings) > 0) {
                    fprintf(stderr,
                            " ** There were warnings when parsing the "
                            "configuration file: %s",
                            config_file);

                    for (int i = 0; i < stringlist_get_size(warnings); i++)
                        fprintf(stderr, " %02d : %s \n", i,
                                stringlist_iget(warnings, i));
                }

                if (config_content_has_item(content, "STDIN"))
                    ext_job_set_stdin_file(
                        ext_job, config_content_iget(content, "STDIN", 0, 0));
                if (config_content_has_item(content, "STDOUT"))
                    ext_job_set_stdout_file(
                        ext_job, config_content_iget(content, "STDOUT", 0, 0));
                else
                    ext_job->stdout_file =
                        util_alloc_filename(NULL, ext_job->name, "stdout");

                if (config_content_has_item(content, "STDERR"))
                    ext_job_set_stderr_file(
                        ext_job, config_content_iget(content, "STDERR", 0, 0));
                else
                    ext_job->stderr_file =
                        util_alloc_filename(NULL, ext_job->name, "stderr");

                if (config_content_has_item(content, "ERROR_FILE"))
                    ext_job_set_error_file(
                        ext_job,
                        config_content_iget(content, "ERROR_FILE", 0, 0));
                if (config_content_has_item(content, "TARGET_FILE"))
                    ext_job_set_target_file(
                        ext_job,
                        config_content_iget(content, "TARGET_FILE", 0, 0));
                if (config_content_has_item(content, "START_FILE"))
                    ext_job_set_start_file(
                        ext_job,
                        config_content_iget(content, "START_FILE", 0, 0));
                if (config_content_has_item(content, "MAX_RUNNING"))
                    ext_job_set_max_running(ext_job,
                                            config_content_iget_as_int(
                                                content, "MAX_RUNNING", 0, 0));
                if (config_content_has_item(content, "MAX_RUNNING_MINUTES"))
                    ext_job_set_max_time(
                        ext_job, config_content_iget_as_int(
                                     content, "MAX_RUNNING_MINUTES", 0, 0));
                if (config_content_has_item(content, MIN_ARG_KEY))
                    ext_job_set_min_arg(
                        ext_job,
                        config_content_iget_as_int(content, MIN_ARG_KEY, 0, 0));
                if (config_content_has_item(content, MAX_ARG_KEY))
                    ext_job_set_max_arg(
                        ext_job,
                        config_content_iget_as_int(content, MAX_ARG_KEY, 0, 0));

                for (int i = 0;
                     i < config_content_get_occurences(content, ARG_TYPE_KEY);
                     i++) {
                    int iarg =
                        config_content_iget_as_int(content, ARG_TYPE_KEY, i, 0);
                    const char *arg_type =
                        config_content_iget(content, ARG_TYPE_KEY, i, 1);

                    ext_job_iset_argtype_string(ext_job, iarg, arg_type);
                }

                char exec_key[20] = EXECUTABLE_KEY;
                bool have_executable =
                    config_content_has_item(content, EXECUTABLE_KEY);
                bool have_portable_exe =
                    config_content_has_item(content, "PORTABLE_EXE");
                if (!have_executable && !have_portable_exe) {
                    fprintf(stderr, "%s: ** '%s' must be set\n", config_file,
                            EXECUTABLE_KEY);
                    ext_job->__valid = false;
                } else if (!have_executable && have_portable_exe) {
                    strcpy(exec_key, "PORTABLE_EXE");
                } else if (have_executable && have_portable_exe) {
                    fprintf(stderr,
                            "%s: ** Ignoring 'PORTABLE_EXE' and using '%s' as "
                            "both were given.\n",
                            config_file, EXECUTABLE_KEY);
                }

                if (ext_job->__valid) {
                    const char *executable =
                        config_content_get_value_as_executable(content,
                                                               exec_key);
                    const char *executable_raw =
                        config_content_iget(content, exec_key, 0, 0);
                    try {
                        ext_job_set_executable(ext_job, executable,
                                               executable_raw, search_path);
                    } catch (std::invalid_argument e) {
                        fprintf(stderr, "Error parsing executable: %s",
                                e.what());
                        ext_job->__valid = false;
                    }
                }

                {
                    if (config_content_has_item(content, "ARGLIST")) {
                        ext_job->deprecated_argv = stringlist_alloc_new();
                        config_content_node_type *arg_node =
                            config_content_get_value_node(content, "ARGLIST");
                        int i;
                        for (i = 0; i < config_content_node_get_size(arg_node);
                             i++)
                            stringlist_append_copy(
                                ext_job->deprecated_argv,
                                config_content_node_iget(arg_node, i));
                    }
                }

                if (config_content_has_item(content, "ENV")) {
                    const config_content_item_type *env_item =
                        config_content_get_item(content, "ENV");
                    for (int ivar = 0;
                         ivar < config_content_item_get_size(env_item);
                         ivar++) {
                        const config_content_node_type *env_node =
                            config_content_item_iget_node(env_item, ivar);
                        const char *key = config_content_node_iget(env_node, 0);
                        if (config_content_node_get_size(env_node) > 1) {
                            const char *value =
                                config_content_node_iget(env_node, 1);
                            hash_insert_hash_owned_ref(
                                ext_job->environment, key,
                                util_alloc_string_copy(value), free);
                        } else
                            hash_insert_ref(ext_job->environment, key, NULL);
                    }
                }

                if (config_content_has_item(content, "EXEC_ENV")) {
                    const config_content_item_type *env_item =
                        config_content_get_item(content, "EXEC_ENV");
                    for (int ivar = 0;
                         ivar < config_content_item_get_size(env_item);
                         ivar++) {
                        const config_content_node_type *env_node =
                            config_content_item_iget_node(env_item, ivar);
                        const char *key = config_content_node_iget(env_node, 0);
                        if (config_content_node_get_size(env_node) > 1) {
                            const char *value =
                                config_content_node_iget(env_node, 1);
                            hash_insert_hash_owned_ref(
                                ext_job->exec_env, key,
                                util_alloc_string_copy(value), free);
                        } else
                            hash_insert_ref(ext_job->exec_env, key, NULL);
                    }
                }

                /* Default mappings; these are used to set values in the argList
           which have not been supplied by the calling context. */
                {
                    if (config_content_has_item(content, "DEFAULT")) {
                        const config_content_item_type *default_item =
                            config_content_get_item(content, "DEFAULT");
                        for (int ivar = 0;
                             ivar < config_content_item_get_size(default_item);
                             ivar++) {
                            const config_content_node_type *default_node =
                                config_content_item_iget_node(default_item,
                                                              ivar);
                            for (int i = 0;
                                 i < config_content_node_get_size(default_node);
                                 i += 2) {
                                const char *key =
                                    config_content_node_iget(default_node, i);
                                const char *value = config_content_node_iget(
                                    default_node, i + 1);
                                hash_insert_hash_owned_ref(
                                    ext_job->default_mapping, key,
                                    util_alloc_string_copy(value), free);
                            }
                        }
                    }
                }

                if (!ext_job->__valid) {
                    // Something NOT OK (i.e. EXECUTABLE now); free the job
                    // instance and return NULL
                    ext_job_free(ext_job);
                    ext_job = NULL;
                    fprintf(stderr,
                            "** Warning: job: \'%s\' not available ... \n",
                            name);
                }
            } else {
                config_error_type *error = config_content_get_errors(content);
                config_error_fprintf(error, true, stderr);
                fprintf(stderr, "** Warning: job: \'%s\' not available ... \n",
                        name);
            }
            config_content_free(content);
        }
        config_free(config);

        return ext_job;
    } else {
        fprintf(stderr,
                "** Warning: you do not have permission to read file:\'%s\' - "
                "job:%s not available. \n",
                config_file, name);
        return NULL;
    }
}

const stringlist_type *ext_job_get_arglist(const ext_job_type *ext_job) {
    if (ext_job->deprecated_argv)
        return ext_job->deprecated_argv;
    else
        return ext_job->argv;
}

const stringlist_type *ext_job_get_argvalues(const ext_job_type *ext_job) {
    stringlist_type *result = stringlist_alloc_new();

    const stringlist_type *argv = ext_job_get_arglist(ext_job);
    for (int i = 0; i < stringlist_get_size(argv); i++) {
        const char *src_string = stringlist_iget(argv, i);
        char *filtered_string = __alloc_filtered_string(
            src_string, ext_job->private_args, ext_job->define_args);
        if (hash_has_key(ext_job->default_mapping, filtered_string))
            filtered_string = (char *)util_realloc_string_copy(
                filtered_string,
                (const char *)hash_get(ext_job->default_mapping,
                                       filtered_string));

        stringlist_append_copy(result, filtered_string);
    }
    return result;
}

/**
   Set the internal arguments of the job based on an input string
   @arg_string which is of the form:

       key1=value1, key2=value2 , key3=value3

   The internal private argument list is cleared before adding these
   arguments.
*/
void ext_job_set_private_args_from_string(ext_job_type *ext_job,
                                          const char *arg_string) {
    subst_list_clear(ext_job->private_args);
    subst_list_add_from_string(ext_job->private_args, arg_string, true);
}

bool ext_job_is_shared(const ext_job_type *ext_job) {
    return !ext_job->private_job;
}

bool ext_job_is_private(const ext_job_type *ext_job) {
    return ext_job->private_job;
}

#undef ASSERT_TOKENS
