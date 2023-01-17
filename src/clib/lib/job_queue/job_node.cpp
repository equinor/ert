#include <filesystem>
#include <string>

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <ert/logging.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/job_node.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("job_queue");

#define JOB_QUEUE_NODE_TYPE_ID 3315299
#define INVALID_QUEUE_INDEX -999

struct job_queue_node_struct {
    /** How many cpu's will this job need - the driver is free to ignore if not relevant. */
    int num_cpu;
    /** The path to the actual executable. */
    char *run_cmd;
    /** The queue will look for the occurence of this file to detect a failure. */
    char *exit_file;
    /** The queue will look for this file to verify that the job is running or
     * has run. */
    char *status_file;
    /** The name of the job. */
    char *job_name;
    /** Where the job is run - absolute path. */
    char *run_path;
    job_callback_ftype *done_callback;
    /** To determine if job can be retried */
    job_callback_ftype *retry_callback;
    /** Callback to perform any cleanup */
    job_callback_ftype *exit_callback;
    void *callback_arg;
    /** The number of commandline arguments to pass when starting the job. */
    int argc;
    /** The commandline arguments. */
    char **argv;
    int queue_index;

    /** Name of the job (in the chain) which has failed. */
    char *failed_job;
    /** The error message from the failed job. */
    char *error_reason;
    char *stderr_capture;
    /** Name of the file containing stderr information. */
    char *stderr_file;

    /** Which attempt is this ... */
    int submit_attempt;
    /** The current status of the job. */
    job_status_type job_status;
    /** Set to true if file status_file has been detected written. */
    bool confirmed_running;
    /** Protecting the access to the job_data pointer. */
    pthread_mutex_t data_mutex;
    /** Driver specific data about this job - fully handled by the driver. */
    void *job_data;
    /** When was the job added to job_queue - the FIRST TIME. */
    time_t submit_time;
    /** When did the job change status -> RUNNING - the LAST TIME. */
    time_t sim_start;
    /** When did the job finish successfully */
    time_t sim_end;
    /** Max waiting between sim_start and confirmed_running is 2 minutes */
    time_t max_confirm_wait;
    /** Timestamp of the status update update file. */
    time_t progress_timestamp;
};

void job_queue_node_free_error_info(job_queue_node_type *node) {
    free(node->error_reason);
    free(node->stderr_capture);
    free(node->stderr_file);
    free(node->failed_job);
}

/*
  When the job script has detected failure it will create a "EXIT"
  file in the runpath directory; this function will inspect the EXIT
  file and determine which job has failed, the reason the job script
  has given to fail the job (typically missing TARGET_FILE) and
  capture the stderr from the job.

  The file is XML formatted:

  ------------------------------------------------
  <error>
     <time>HH:MM:SS</time>
     <job> Name of job </job>
     <reason> Reason why the job failed </reason>
     <stderr>
        Capture of stderr from the job, can typically be
        a multiline string.
     </stderr>
  </error>
  ------------------------------------------------

  This format is written by the dump_EXIT_file() function in the
  job_dispatch.py script.
*/

/*
   This extremely half-assed XML "parsing" should of course be kept a
   secret...
*/

static char *__alloc_tag_content(const char *xml_buffer, const char *tag) {
    char *open_tag = (char *)util_alloc_sprintf("<%s>", tag);
    char *close_tag = (char *)util_alloc_sprintf("</%s>", tag);

    const char *start_ptr = strstr(xml_buffer, open_tag);
    const char *end_ptr = strstr(xml_buffer, close_tag);
    char *tag_content = NULL;

    if ((start_ptr != NULL) && (end_ptr != NULL)) {
        int length;
        start_ptr += strlen(open_tag);

        length = end_ptr - start_ptr;
        tag_content = util_alloc_substring_copy(start_ptr, 0, length);
    }

    free(open_tag);
    free(close_tag);
    return tag_content;
}

/**
   This code is meant to capture which of the jobs has failed; why it
   has failed and the stderr stream of the failing job. Depending on
   the failure circumstances the EXIT file might not be around.
*/
void job_queue_node_fscanf_EXIT(job_queue_node_type *node) {
    job_queue_node_free_error_info(node);
    if (node->exit_file) {
        if (fs::exists(node->exit_file)) {
            char *xml_buffer =
                util_fread_alloc_file_content(node->exit_file, NULL);

            node->failed_job = __alloc_tag_content(xml_buffer, "job");
            node->error_reason = __alloc_tag_content(xml_buffer, "reason");
            node->stderr_capture = __alloc_tag_content(xml_buffer, "stderr");
            node->stderr_file = __alloc_tag_content(xml_buffer, "stderr_file");

            free(xml_buffer);
        } else
            node->failed_job = util_alloc_sprintf(
                "EXIT file:%s not found - load failure?", node->exit_file);
    }
}

int job_queue_node_get_queue_index(const job_queue_node_type *node) {
    if (node->queue_index == INVALID_QUEUE_INDEX)
        util_abort("%s: internal error: asked for not-yet-initialized "
                   "node->queue_index\n",
                   __func__);
    return node->queue_index;
}

void job_queue_node_set_queue_index(job_queue_node_type *node,
                                    int queue_index) {
    if (node->queue_index == INVALID_QUEUE_INDEX)
        node->queue_index = queue_index;
    else
        util_abort("%s: internal error: attempt to reset queue_index \n",
                   __func__);
}

/*
 The error information is retained even after the job has completed
 completely, so that calling scope can ask for it - that is the
 reason there are separate free() and clear functions for the error related fields.
*/

void job_queue_node_free_data(job_queue_node_type *node) {
    free(node->job_name);
    free(node->exit_file);
    free(node->status_file);
    free(node->run_cmd);
    util_free_stringlist(node->argv, node->argc);
}

void job_queue_node_free(job_queue_node_type *node) {
    // The callback_arg will not be freed; that will be the responsability of
    // the calling scope.
    job_queue_node_free_data(node);
    job_queue_node_free_error_info(node);
    free(node->run_path);
    free(node);
}

job_status_type job_queue_node_get_status(const job_queue_node_type *node) {
    return node->job_status;
}

int job_queue_node_get_submit_attempt(const job_queue_node_type *node) {
    return node->submit_attempt;
}

job_queue_node_type *job_queue_node_alloc_simple(const char *job_name,
                                                 const char *run_path,
                                                 const char *run_cmd, int argc,
                                                 const char **argv) {
    return job_queue_node_alloc(job_name, run_path, run_cmd, argc, argv, 1,
                                NULL, NULL, NULL, NULL, NULL, NULL);
}

job_queue_node_type *
job_queue_node_alloc_python(const char *job_name, const char *run_path,
                            const char *run_cmd, int argc,
                            const stringlist_type *arguments, int num_cpu,
                            const char *status_file, const char *exit_file) {
    char **argv = stringlist_alloc_char_ref(arguments);
    job_queue_node_type *out =
        job_queue_node_alloc(job_name, run_path, run_cmd, argc, argv, num_cpu,
                             status_file, exit_file, NULL, NULL, NULL, NULL);
    free(argv);
    return out;
}

job_queue_node_type *
job_queue_node_alloc(const char *job_name, const char *run_path,
                     const char *run_cmd, int argc, char const *const *argv,
                     int num_cpu, const char *status_file,
                     const char *exit_file, job_callback_ftype *done_callback,
                     job_callback_ftype *retry_callback,
                     job_callback_ftype *exit_callback, void *callback_arg) {

    if (!util_is_directory(run_path))
        return NULL;

    job_queue_node_type *node =
        (job_queue_node_type *)util_malloc(sizeof *node);
    node->confirmed_running = false;
    node->progress_timestamp = time(NULL);

    /* The data initialized in this block should *NEVER* change. */
    std::string path = job_name;
    std::string basename = path.substr(path.find_last_of("/\\") + 1);
    node->job_name = util_alloc_string_copy(basename.data());

    node->run_path = util_alloc_realpath(run_path);

    node->run_cmd = util_alloc_string_copy(run_cmd);
    node->argc = argc;
    node->argv = util_alloc_stringlist_copy(
        argv, argc); // Please fix const <type> ** in libecl
    node->num_cpu = num_cpu;

    if (status_file)
        node->status_file =
            util_alloc_filename(node->run_path, status_file, NULL);
    else
        node->status_file = NULL;

    if (exit_file)
        node->exit_file = util_alloc_filename(node->run_path, exit_file, NULL);
    else
        node->exit_file = NULL;

    node->exit_callback = exit_callback;
    node->retry_callback = retry_callback;
    node->done_callback = done_callback;
    node->callback_arg = callback_arg;

    node->error_reason = NULL;
    node->stderr_capture = NULL;
    node->stderr_file = NULL;
    node->failed_job = NULL;

    node->job_status = JOB_QUEUE_NOT_ACTIVE;
    node->queue_index = INVALID_QUEUE_INDEX;
    node->submit_attempt = 0;
    node->job_data = NULL; // assume allocation is run in single thread mode
    node->sim_start = 0;
    node->sim_end = 0;
    node->submit_time = time(NULL);
    node->max_confirm_wait = 60 * 2; // 2 minutes before we consider job dead.

    pthread_mutex_init(&node->data_mutex, NULL);
    return node;
}

double job_queue_node_time_since_sim_start(const job_queue_node_type *node) {
    return util_difftime_seconds(node->sim_start, time(NULL));
}

void job_queue_node_set_status(job_queue_node_type *node,
                               job_status_type new_status) {
    if (new_status == node->job_status)
        return;

    logger->debug("Set {}({}) to {}", node->job_name, node->queue_index,
                  job_status_get_name(new_status));
    node->job_status = new_status;

    // We record sim start when the node is in state JOB_QUEUE_WAITING to be
    // sure that we do not miss the start time completely for very fast jobs
    // which are registered in the state JOB_QUEUE_RUNNING.
    if (new_status == JOB_QUEUE_WAITING)
        node->sim_start = time(NULL);

    if (new_status == JOB_QUEUE_RUNNING)
        node->sim_start = time(NULL);

    if (!(new_status & JOB_QUEUE_COMPLETE_STATUS))
        return;

    node->sim_end = time(NULL);
    node->progress_timestamp = node->sim_end;

    if (new_status == JOB_QUEUE_FAILED)
        job_queue_node_fscanf_EXIT(node);
}

submit_status_type job_queue_node_submit(job_queue_node_type *node,
                                         job_queue_status_type *status,
                                         queue_driver_type *driver) {
    submit_status_type submit_status;
    pthread_mutex_lock(&node->data_mutex);

    void *job_data = queue_driver_submit_job(
        driver, node->run_cmd, node->num_cpu, node->run_path, node->job_name,
        node->argc, (const char **)node->argv);
    job_status_type old_status;
    job_status_type new_status;

    if (job_data == NULL) {
        // In this case the status of the job itself will be
        // unmodified; i.e. it will still be WAITING, and a new attempt
        // to submit it will be performed in the next round.
        submit_status = SUBMIT_DRIVER_FAIL;
        logger->warning("Failed to submit job {} (attempt {})", node->job_name,
                        node->submit_attempt);
        goto cleanup;
    }

    old_status = node->job_status;
    new_status = JOB_QUEUE_SUBMITTED;

    logger->info("Submitted job {} (attempt {})", node->job_name,
                 node->submit_attempt);

    node->job_data = job_data;
    node->submit_attempt++;
    // The status JOB_QUEUE_SUBMITTED is internal, and not exported anywhere.
    // The job_queue_update_status() will update this to PENDING or RUNNING at
    // the next call. The important difference between SUBMITTED and WAITING is
    // that SUBMITTED have job_data != NULL and the job_queue_node free
    // function must be called on it.
    submit_status = SUBMIT_OK;
    job_queue_node_set_status(node, new_status);
    job_queue_status_transition(status, old_status, new_status);

cleanup:
    pthread_mutex_unlock(&node->data_mutex);
    return submit_status;
}

submit_status_type job_queue_node_submit_simple(job_queue_node_type *node,
                                                queue_driver_type *driver) {
    submit_status_type submit_status;
    pthread_mutex_lock(&node->data_mutex);
    job_queue_node_set_status(node, JOB_QUEUE_SUBMITTED);
    void *job_data = queue_driver_submit_job(
        driver, node->run_cmd, node->num_cpu, node->run_path, node->job_name,
        node->argc, (const char **)node->argv);
    job_status_type old_status;
    job_status_type new_status;

    if (job_data == NULL) {
        // In this case the status of the job itself will be
        // unmodified; i.e. it will still be WAITING, and a new attempt
        // to submit it will be performed in the next round.
        submit_status = SUBMIT_DRIVER_FAIL;
        logger->warning("Failed to submit job {} (attempt {})", node->job_name,
                        node->submit_attempt);
        pthread_mutex_unlock(&node->data_mutex);
        return submit_status;
    }

    old_status = node->job_status;
    new_status = JOB_QUEUE_SUBMITTED;

    logger->info("Submitted job {} (attempt {})", node->job_name,
                 node->submit_attempt);

    node->job_data = job_data;
    node->submit_attempt++;
    // The status JOB_QUEUE_SUBMITTED is internal, and not exported anywhere.
    // The job_queue_update_status() will update this to PENDING or RUNNING at
    // the next call. The important difference between SUBMITTED and WAITING is
    // that SUBMITTED have job_data != NULL and the job_queue_node free
    // function must be called on it.
    submit_status = SUBMIT_OK;
    job_queue_node_set_status(node, new_status);
    pthread_mutex_unlock(&node->data_mutex);
    return submit_status;
}

static bool
job_queue_node_status_update_confirmed_running__(job_queue_node_type *node) {
    if (node->confirmed_running)
        return true;

    if (!node->status_file) {
        node->confirmed_running = true;
        return true;
    }

    if (fs::exists(node->status_file))
        node->confirmed_running = true;
    return node->confirmed_running;
}

static void job_queue_node_update_timestamp(job_queue_node_type *node) {
    if (node->job_status != JOB_QUEUE_RUNNING)
        return;

    if (!node->status_file)
        return;

    time_t mtime = util_file_mtime(node->status_file);
    if (mtime > 0)
        node->progress_timestamp = mtime;
}

/**
if status = running, and current_time > sim_start + max_confirm_wait
(usually 2 min), check if job is confirmed running (status_file exists).
If not confirmed, set job to JOB_QUEUE_FAILED.
*/
bool job_queue_node_update_status(job_queue_node_type *node,
                                  job_queue_status_type *status,
                                  queue_driver_type *driver) {
    bool status_change = false;
    pthread_mutex_lock(&node->data_mutex);

    job_status_type current_status;
    bool confirmed;

    if (!node->job_data)
        goto cleanup;

    current_status = job_queue_node_get_status(node);

    confirmed = job_queue_node_status_update_confirmed_running__(node);

    if ((current_status & JOB_QUEUE_RUNNING) && !confirmed) {
        // it's running, but not confirmed running.
        double runtime = job_queue_node_time_since_sim_start(node);
        if (runtime >= node->max_confirm_wait) {
            logger->info("max_confirm_wait ({}) has passed since sim_start"
                         "without success; {} is dead (attempt {})",
                         node->max_confirm_wait, node->job_name,
                         node->submit_attempt);
            job_status_type new_status = JOB_QUEUE_DO_KILL_NODE_FAILURE;
            status_change =
                job_queue_status_transition(status, current_status, new_status);
            job_queue_node_set_status(node, new_status);
        }
    }

    current_status = job_queue_node_get_status(node);
    if (current_status & JOB_QUEUE_CAN_UPDATE_STATUS) {
        job_status_type new_status =
            queue_driver_get_status(driver, node->job_data);
        status_change =
            job_queue_status_transition(status, current_status, new_status);
        job_queue_node_set_status(node, new_status);
    }

cleanup:
    job_queue_node_update_timestamp(node);
    pthread_mutex_unlock(&node->data_mutex);
    return status_change;
}

job_status_type job_queue_node_refresh_status(job_queue_node_type *node,
                                              queue_driver_type *driver) {
    pthread_mutex_lock(&node->data_mutex);
    job_status_type current_status = job_queue_node_get_status(node);
    bool confirmed;

    if (!node->job_data) {
        job_queue_node_update_timestamp(node);
        pthread_mutex_unlock(&node->data_mutex);
        return current_status;
    }

    confirmed = job_queue_node_status_update_confirmed_running__(node);

    if ((current_status & JOB_QUEUE_RUNNING) && !confirmed) {
        // it's running, but not confirmed running.
        double runtime = job_queue_node_time_since_sim_start(node);
        if (runtime >= node->max_confirm_wait) {
            logger->info("max_confirm_wait ({}) has passed since sim_start"
                         "without success; {} is dead (attempt {})",
                         node->max_confirm_wait, node->job_name,
                         node->submit_attempt);
            job_status_type new_status = JOB_QUEUE_DO_KILL_NODE_FAILURE;
            job_queue_node_set_status(node, new_status);
        }
    }

    current_status = job_queue_node_get_status(node);
    if (current_status & JOB_QUEUE_CAN_UPDATE_STATUS) {
        job_status_type new_status =
            queue_driver_get_status(driver, node->job_data);
        job_queue_node_set_status(node, new_status);
        current_status = job_queue_node_get_status(node);
    }
    pthread_mutex_unlock(&node->data_mutex);
    return current_status;
}

bool job_queue_node_status_transition(job_queue_node_type *node,
                                      job_queue_status_type *status,
                                      job_status_type new_status) {
    bool status_change = false;
    pthread_mutex_lock(&node->data_mutex);

    job_status_type old_status = job_queue_node_get_status(node);
    status_change = job_queue_status_transition(status, old_status, new_status);

    if (status_change)
        job_queue_node_set_status(node, new_status);

    pthread_mutex_unlock(&node->data_mutex);
    return status_change;
}

bool job_queue_node_kill(job_queue_node_type *node,
                         job_queue_status_type *status,
                         queue_driver_type *driver) {
    bool result = false;
    pthread_mutex_lock(&node->data_mutex);

    job_status_type current_status = job_queue_node_get_status(node);
    if (current_status & JOB_QUEUE_CAN_KILL) {
        // If the job is killed before it is even started no driver specific
        // job data has been assigned; we therefore must check the
        // node->job_data pointer before entering.
        if (node->job_data) {
            queue_driver_kill_job(driver, node->job_data);
            queue_driver_free_job(driver, node->job_data);
            node->job_data = NULL;
        }
        job_queue_status_transition(status, current_status,
                                    JOB_QUEUE_IS_KILLED);
        job_queue_node_set_status(node, JOB_QUEUE_IS_KILLED);
        logger->info("Job {} set to killed", node->job_name);
        result = true;
    } else {
        logger->warning("node_kill called but cannot kill {}", node->job_name);
    }

    pthread_mutex_unlock(&node->data_mutex);
    return result;
}

bool job_queue_node_kill_simple(job_queue_node_type *node,
                                queue_driver_type *driver) {
    bool result = false;
    pthread_mutex_lock(&node->data_mutex);
    job_status_type current_status = job_queue_node_get_status(node);
    if (current_status & JOB_QUEUE_CAN_KILL) {
        // If the job is killed before it is even started no driver specific
        // job data has been assigned; we therefore must check the
        // node->job_data pointer before entering.
        if (node->job_data) {
            queue_driver_kill_job(driver, node->job_data);
            queue_driver_free_job(driver, node->job_data);
            node->job_data = NULL;
        }
        job_queue_node_set_status(node, JOB_QUEUE_IS_KILLED);
        logger->info("job {} set to killed", node->job_name);
        result = true;
    } else {
        logger->warning("node_kill called but cannot kill {}", node->job_name);
    }
    pthread_mutex_unlock(&node->data_mutex);
    return result;
}

/**
  This returns a pointer to a very internal datastructure; used by the
  Job class in Python which interacts directly with the driver
  implementation. This is too low level, and the whole Driver / Job
  implementation in Python should be changed to only expose the higher
  level queue class.
*/
void *job_queue_node_get_driver_data(job_queue_node_type *node) {
    return node->job_data;
}
