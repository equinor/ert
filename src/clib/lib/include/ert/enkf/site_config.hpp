#ifndef ERT_SITE_CONFIG_H
#define ERT_SITE_CONFIG_H

#include <stdbool.h>

#include <ert/util/stringlist.h>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/job_queue.hpp>

typedef struct site_config_struct site_config_type;

extern "C" const char *site_config_get_location();
extern "C" const char *site_config_get_config_file(const site_config_type *);
extern "C" PY_USED const char *
site_config_get_license_root_path(const site_config_type *site_config);
extern "C" void
site_config_set_license_root_path(site_config_type *site_config,
                                  const char *license_root_path);
extern "C" void site_config_free(site_config_type *);
extern "C" ext_joblist_type *
site_config_get_installed_jobs(const site_config_type *);
extern "C" const env_varlist_type *
site_config_get_env_varlist(const site_config_type *site_config);
int site_config_install_job(site_config_type *site_config, const char *job_name,
                            const char *install_file);
extern "C" site_config_type *site_config_alloc_default();
extern "C" site_config_type *
site_config_alloc(const config_content_type *config_content);
extern "C" site_config_type *
site_config_alloc_full(ext_joblist_type *ext_joblist,
                       env_varlist_type *env_varlist);
config_content_type *site_config_alloc_content(config_parser_type *);
#endif
