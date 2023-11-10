#pragma once
#include <ert/job_queue/queue_driver.hpp>
#include <string>
#include <vector>
/*
  The options supported by the LSF driver.
*/
#define LSF_QUEUE "LSF_QUEUE"
#define LSF_RESOURCE "LSF_RESOURCE"
#define LSF_SERVER "LSF_SERVER"
#define LSF_RSH_CMD                                                            \
    "LSF_RSH_CMD" // This option is set to DEFAULT_RSH_CMD at driver creation.
#define LSF_LOGIN_SHELL "LSF_LOGIN_SHELL" // Not fully implemented yet
#define LSF_BSUB_CMD "BSUB_CMD"
#define LSF_BJOBS_CMD "BJOBS_CMD"
#define LSF_BKILL_CMD "BKILL_CMD"
#define LSF_BHIST_CMD "BHIST_CMD"
#define LSF_BJOBS_TIMEOUT "BJOBS_TIMEOUT"
#define LSF_DEBUG_OUTPUT "DEBUG_OUTPUT"
#define LSF_SUBMIT_SLEEP "SUBMIT_SLEEP"
#define LSF_EXCLUDE_HOST "EXCLUDE_HOST"
#define LSF_PROJECT_CODE "PROJECT_CODE"

#define LOCAL_LSF_SERVER "LOCAL"
#define NULL_LSF_SERVER "NULL"
#define DEFAULT_SUBMIT_SLEEP "0"

#define JOB_STAT_NULL 0
#define JOB_STAT_PEND 1
#define JOB_STAT_SSUSP 0x08
#define JOB_STAT_USUSP 0x10
#define JOB_STAT_PSUSP 0x02
#define JOB_STAT_RUN 0x04
#define JOB_STAT_EXIT 0x20
#define JOB_STAT_DONE 0x40
#define JOB_STAT_PDONE 0x80
#define JOB_STAT_UNKWN 0x10000

typedef enum {
    LSF_SUBMIT_INVALID = 0,
    LSF_SUBMIT_LOCAL_SHELL = 2,
    LSF_SUBMIT_REMOTE_SHELL = 3
} lsf_submit_method_enum;

typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct lsf_job_type;

const std::vector<std::string> LSF_DRIVER_OPTIONS = {
    LSF_QUEUE,        LSF_RESOURCE,      LSF_SERVER,       LSF_RSH_CMD,
    LSF_LOGIN_SHELL,  LSF_BSUB_CMD,      LSF_BJOBS_CMD,    LSF_BKILL_CMD,
    LSF_BHIST_CMD,    LSF_BJOBS_TIMEOUT, LSF_DEBUG_OUTPUT, LSF_SUBMIT_SLEEP,
    LSF_EXCLUDE_HOST, LSF_PROJECT_CODE};

void lsf_job_free(lsf_job_type *job);

void *lsf_driver_alloc();
void *lsf_driver_submit_job(void *_driver, std::string submit_cmd, int num_cpu,
                            fs::path run_path, std::string job_name);
job_status_type lsf_driver_convert_status(int lsf_status);
void lsf_driver_kill_job(void *_driver, void *_job);
void lsf_driver_free_(void *_driver);
void lsf_driver_free(lsf_driver_type *driver);
job_status_type lsf_driver_get_job_status(void *_driver, void *_job);
int lsf_driver_get_job_status_lsf(void *_driver, void *_job);
void lsf_driver_free_job(void *_job);
void lsf_driver_set_bjobs_refresh_interval(lsf_driver_type *driver,
                                           int refresh_interval);

void lsf_driver_add_exclude_hosts(lsf_driver_type *driver,
                                  const char *excluded);
lsf_submit_method_enum
lsf_driver_get_submit_method(const lsf_driver_type *driver);

const void *lsf_driver_get_option(const void *_driver, const char *option_key);
bool lsf_driver_set_option(void *_driver, const char *option_key,
                           const void *value);
bool lsf_driver_has_project_code(const lsf_driver_type *driver);
int lsf_job_parse_bsub_stdout(std::string);
