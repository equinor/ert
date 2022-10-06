#include <filesystem>
#include <string>

#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ert/util/stringlist.h>
#include <ert/util/util.h>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/job_queue.hpp>

#include <ert/config/config_keywords.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/site_config.hpp>

namespace fs = std::filesystem;

/**
   This struct contains information which is specific to the site
   where this enkf instance is running. Pointers to the fields in this
   structure are passed on to e.g. the enkf_state->shared_info object,
   but this struct is the *OWNER* of this information, and hence
   responsible for booting and deleting these objects.

   The settings held by the site_config object are by default set in
   the site-wide configuration file, but they can also be overridden
   in the users configuration file. This makes both parsing,
   validating and also storing the configuration information a bit
   more tricky:

   Parsing:
   --------
   When parsing the user configuration file all settings are optional,
   that means that the required validation of the config system, can
   not be used, instead every get must be preceeded by:

      if (config_content_has_item(config , KEY)) ...

   Furthermore everything is done twice; first with config as a
   site-config instance, and later as user-config instance.


   Saving:
   -------
   A setting which originates from the site_config file should not be
   stored in the user's config file, but additions/overrides from the
   user's config file should of course be saved. This is 'solved' with
   many fields having a xxx_site duplicate, where the xxx_site is only
   updated during the initial parsing of the site-config file; when
   the flag user_mode is set to true the xxx_site fields are not
   updated. When saving only fields which are different from their
   xxx_site counterpart are stored.
 */
struct site_config_struct {

    char *config_file;

    /** The list of external jobs which have been installed. These jobs will be
     * the parts of the forward model. */
    ext_joblist_type *joblist;

    /** Container for the environment variables set in the user config file. */
    env_varlist_type *env_varlist;

    /** The license_root_path value set by the user. */
    char *license_root_path;
    /** The license_root_path value set by the site. */
    char *license_root_path_site;
    /** The license_root_path value actually used - includes a user/pid subdirectory. */
    char *__license_root_path;

    bool user_mode;
    bool search_path;
};

static bool site_config_init(site_config_type *site_config,
                             const config_content_type *config);
static void site_config_init_env(site_config_type *site_config,
                                 const config_content_type *config);

static void site_config_set_config_file(site_config_type *site_config,
                                        const char *config_file) {
    free(site_config->config_file);
    site_config->config_file =
        util_realloc_string_copy(site_config->config_file, config_file);
}

/**
   This site_config object is not really ready for prime time.
*/
static site_config_type *site_config_alloc_empty() {
    site_config_type *site_config =
        (site_config_type *)util_malloc(sizeof *site_config);

    site_config->joblist = ext_joblist_alloc();

    site_config->config_file = NULL;
    site_config->license_root_path = NULL;
    site_config->license_root_path_site = NULL;
    site_config->__license_root_path = NULL;
    site_config->user_mode = false;

    site_config->env_varlist = env_varlist_alloc();

    site_config->search_path = false;
    return site_config;
}

static void site_config_load_config(site_config_type *site_config) {
    config_parser_type *config = config_alloc();
    config_content_type *content = site_config_alloc_content(config);

    site_config_init(site_config, content);

    config_free(config);
    config_content_free(content);
}

/**
 * NOTE: The queue config is not loaded until the site_config_alloc_load_user.
 */
site_config_type *site_config_alloc_default() {
    site_config_type *site_config = site_config_alloc_empty();
    site_config_set_config_file(site_config, site_config_get_location());
    site_config_load_config(site_config);

    return site_config;
}

site_config_type *site_config_alloc(const config_content_type *config_content) {
    site_config_type *site_config = site_config_alloc_default();

    if (config_content) {
        site_config->user_mode = true;
        site_config_init(site_config, config_content);
    }

    return site_config;
}

site_config_type *site_config_alloc_full(ext_joblist_type *ext_joblist,
                                         env_varlist_type *env_varlist) {
    site_config_type *site_config = site_config_alloc_empty();
    site_config->joblist = ext_joblist;
    site_config->env_varlist = env_varlist;
    return site_config;
}

const char *
site_config_get_license_root_path(const site_config_type *site_config) {
    return site_config->license_root_path;
}

/**
   Observe that this variable can not "really" be set to different
   values during a simulation, when creating ext_job instances they
   will store a pointer to this variable on creation, if the variable
   is later changed they will be left with a dangling copy. That is
   not particularly elegant, however it should nonetheless work.
*/
void site_config_set_license_root_path(site_config_type *site_config,
                                       const char *license_root_path) {
    util_make_path(license_root_path);
    {
        char *full_license_root_path = util_alloc_realpath(license_root_path);
        {
            // Appending /user/pid to the license root path. Everything
            // including the pid is removed when exiting (gracefully ...).

            // Dangling license directories after a crash can just be removed.
            site_config->license_root_path = util_realloc_string_copy(
                site_config->license_root_path, full_license_root_path);
            site_config->__license_root_path = util_realloc_sprintf(
                site_config->__license_root_path, "%s%c%s%c%d",
                full_license_root_path, UTIL_PATH_SEP_CHAR, getenv("USER"),
                UTIL_PATH_SEP_CHAR, getpid());

            if (!site_config->user_mode)
                site_config->license_root_path_site = util_realloc_string_copy(
                    site_config->license_root_path_site,
                    full_license_root_path);
        }
        free(full_license_root_path);
    }
}

/**
   Will return 0 if the job is added correctly, and a non-zero (not
   documented ...) error code if the job is not added.
*/
int site_config_install_job(site_config_type *site_config, const char *job_name,
                            const char *install_file) {
    ext_job_type *new_job = ext_job_fscanf_alloc(
        job_name, site_config->__license_root_path, site_config->user_mode,
        install_file, site_config->search_path);
    if (new_job != NULL) {
        ext_joblist_add_job(site_config->joblist, job_name, new_job);
        return 0;
    } else
        return 1; /* Some undocumented error condition - the job is NOT added. */
}

static void site_config_add_jobs(site_config_type *site_config,
                                 const config_content_type *config) {
    if (config_content_has_item(config, INSTALL_JOB_KEY)) {
        const config_content_item_type *content_item =
            config_content_get_item(config, INSTALL_JOB_KEY);
        int num_jobs = config_content_item_get_size(content_item);
        for (int job_nr = 0; job_nr < num_jobs; job_nr++) {
            config_content_node_type *node =
                config_content_item_iget_node(content_item, job_nr);
            const char *job_key = config_content_node_iget(node, 0);
            const char *description_file =
                config_content_node_iget_as_abspath(node, 1);

            site_config_install_job(site_config, job_key, description_file);
        }
    }
    if (config_content_has_item(config, INSTALL_JOB_DIRECTORY_KEY)) {
        const config_content_item_type *content_item =
            config_content_get_item(config, INSTALL_JOB_DIRECTORY_KEY);
        int num_dirs = config_content_item_get_size(content_item);
        for (int dir_nr = 0; dir_nr < num_dirs; dir_nr++) {
            config_content_node_type *node =
                config_content_item_iget_node(content_item, dir_nr);
            const char *directory =
                config_content_node_iget_as_abspath(node, 0);

            ext_joblist_add_jobs_in_directory(site_config->joblist, directory,
                                              site_config->__license_root_path,
                                              site_config->user_mode,
                                              site_config->search_path);
        }
    }
}

const env_varlist_type *
site_config_get_env_varlist(const site_config_type *site_config) {
    return site_config->env_varlist;
}

static void site_config_init_env(site_config_type *site_config,
                                 const config_content_type *config) {
    {
        if (config_content_has_item(config, SETENV_KEY)) {
            config_content_item_type *setenv_item =
                config_content_get_item(config, SETENV_KEY);
            int i;
            for (i = 0; i < config_content_item_get_size(setenv_item); i++) {
                const config_content_node_type *setenv_node =
                    config_content_item_iget_node(setenv_item, i);
                const char *var = config_content_node_iget(setenv_node, 0);
                const char *value = config_content_node_iget(setenv_node, 1);

                env_varlist_setenv(site_config->env_varlist, var, value);
            }
        }
    }

    {
        if (config_content_has_item(config, UPDATE_PATH_KEY)) {
            config_content_item_type *path_item =
                config_content_get_item(config, UPDATE_PATH_KEY);
            int i;
            for (i = 0; i < config_content_item_get_size(path_item); i++) {
                const config_content_node_type *path_node =
                    config_content_item_iget_node(path_item, i);
                const char *path = config_content_node_iget(path_node, 0);
                const char *value = config_content_node_iget(path_node, 1);

                env_varlist_update_path(site_config->env_varlist, path, value);
            }
        }
    }
}

/**
   This function will be called twice, first when the config instance
   is an internalization of the site-wide configuration file, and
   secondly when config is an internalisation of the user's
   configuration file. The @user_config parameter will be true in the
   latter case.
 */
static bool site_config_init(site_config_type *site_config,
                             const config_content_type *config) {

    site_config_add_jobs(site_config, config);
    site_config_init_env(site_config, config);

    if (config_content_has_item(config, LICENSE_PATH_KEY))
        site_config_set_license_root_path(
            site_config,
            config_content_get_value_as_abspath(config, LICENSE_PATH_KEY));

    return true;
}

void site_config_free(site_config_type *site_config) {
    ext_joblist_free(site_config->joblist);

    env_varlist_free(site_config->env_varlist);

    if (site_config->__license_root_path != NULL)
        util_clear_directory(site_config->__license_root_path, true, true);

    free(site_config->config_file);

    free(site_config->license_root_path);
    free(site_config->license_root_path_site);
    free(site_config->__license_root_path);

    free(site_config);
}

ext_joblist_type *
site_config_get_installed_jobs(const site_config_type *site_config) {
    return site_config->joblist;
}

const char *site_config_get_config_file(const site_config_type *site_config) {
    return site_config->config_file;
}

config_content_type *
site_config_alloc_content(config_parser_type *config_parser) {

    const char *site_config_file = site_config_get_location();

    if (site_config_file == NULL)
        util_abort("%s: No config file specified.\n", __func__);

    if (!fs::exists(site_config_file))
        util_abort("%s: can not locate site configuration file:%s \n", __func__,
                   site_config_file);

    init_site_config_parser(config_parser);
    config_content_type *content =
        config_parse(config_parser, site_config_file, "--", INCLUDE_KEY,
                     DEFINE_KEY, NULL, CONFIG_UNRECOGNIZED_WARN, true);

    if (!config_content_is_valid(content)) {
        config_error_type *errors = config_content_get_errors(content);
        fprintf(stderr,
                "** ERROR: Parsing site configuration file:%s failed \n\n",
                site_config_file);
        config_error_fprintf(errors, true, stderr);
        util_abort("%s: Invalid configurations in site_config file: %s.\n",
                   __func__, site_config_file);
    }

    return content;
}

static std::string _site_config;

extern "C" void set_site_config(const char *site_config) {
    auto path = realpath(site_config, NULL);
    if (!path) {
        perror("Failed to set default site config file");
    } else {
        _site_config = path;
        free(path);
    }
}

const char *site_config_get_location() {
    const char *site_config = _site_config.c_str();
    const char *env_site_config = getenv("ERT_SITE_CONFIG");

    if (env_site_config != NULL) {
        if (fs::exists(env_site_config)) {
            site_config = env_site_config;
        } else {
            fprintf(stderr,
                    "The environment variable ERT_SITE_CONFIG points to "
                    "non-existing file: %s - ignored\n",
                    env_site_config);
        }
    }

    if (site_config == NULL) {
        fprintf(stderr,
                "**WARNING** main enkf_config file is not set. Use environment "
                "variable \"ERT_SITE_CONFIG\" - or recompile.\n");
    }

    return site_config;
}
