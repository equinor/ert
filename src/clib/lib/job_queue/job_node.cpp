#include <filesystem>
#include <string>

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tuple>

#include <ert/logging.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/job_node.hpp>
#include <ert/python.hpp>
#include <fmt/format.h>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("job_queue");

#define INVALID_QUEUE_INDEX -999
const time_t MAX_CONFIRMED_WAIT = 10 * 60;

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

static std::string __alloc_tag_content(const char *xml_buffer,
                                       const char *tag) {
    char *open_tag = (char *)util_alloc_sprintf("<%s>", tag);
    char *close_tag = (char *)util_alloc_sprintf("</%s>", tag);

    const char *start_ptr = strstr(xml_buffer, open_tag);
    const char *end_ptr = strstr(xml_buffer, close_tag);
    std::string tag_content = "";

    if ((start_ptr != NULL) && (end_ptr != NULL)) {
        start_ptr += strlen(open_tag);

        int length = end_ptr - start_ptr;
        char *substr = util_alloc_substring_copy(start_ptr, 0, length);
        tag_content = std::string(substr);
        free(substr);
    }

    free(open_tag);
    free(close_tag);
    return tag_content;
}

std::string __add_tabs(std::string incoming) {
    std::string incoming_tabbed = "";
    std::string incoming_line = "";
    std::stringstream incoming_stream(incoming);
    while (std::getline(incoming_stream, incoming_line, '\n')) {
        incoming_tabbed += "\t" + incoming_line + "\n";
    }
    incoming_tabbed.pop_back();
    return incoming_tabbed;
}

/**
   This code is meant to capture which of the jobs has failed; why it
   has failed and the stderr stream of the failing job. Depending on
   the failure circumstances the EXIT file might not be around.
*/
void job_queue_node_fscanf_EXIT(job_queue_node_type *node) {
    if (node->exit_file) {
        if (!fs::exists(node->exit_file)) {
            node->fail_message =
                fmt::format("EXIT file:{} not found", node->exit_file);
            return;
        }
        char *xml_buffer = util_fread_alloc_file_content(node->exit_file, NULL);

        std::string failed_job = __alloc_tag_content(xml_buffer, "job");
        std::string error_reason = __alloc_tag_content(xml_buffer, "reason");
        std::string stderr_file =
            __alloc_tag_content(xml_buffer, "stderr_file");
        std::string stderr_capture =
            __add_tabs(__alloc_tag_content(xml_buffer, "stderr"));
        node->fail_message = fmt::format(
            "job {} failed with: '{}'\n\tstderr file: '{}',\n\tits contents:{}",
            failed_job, error_reason, stderr_file, stderr_capture);

        free(xml_buffer);
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
    job_queue_node_free_data(node);
    free(node->run_path);
    delete node;
}

job_status_type job_queue_node_get_status(const job_queue_node_type *node) {
    return node->job_status;
}

job_queue_node_type *job_queue_node_alloc(const char *job_name,
                                          const char *run_path,
                                          const char *run_cmd, int argc,
                                          const stringlist_type *arguments,
                                          int num_cpu, const char *status_file,
                                          const char *exit_file) {
    char **argv = stringlist_alloc_char_ref(arguments);
    if (!util_is_directory(run_path))
        return NULL;

    auto node = new job_queue_node_type;

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

    node->job_status = JOB_QUEUE_NOT_ACTIVE;
    node->queue_index = INVALID_QUEUE_INDEX;
    node->submit_attempt = 0;
    node->job_data = NULL; // assume allocation is run in single thread mode
    node->sim_start = 0;

    pthread_mutex_init(&node->data_mutex, NULL);
    free(argv);
    return node;
}

void job_queue_node_set_status(job_queue_node_type *node,
                               job_status_type new_status) {
    if (new_status == node->job_status)
        return;

    logger->debug("Set {}({}) to {}", node->job_name, node->queue_index,
                  job_status_names.at(new_status).c_str());
    node->job_status = new_status;

    // We record sim start when the node is in state JOB_QUEUE_WAITING to be
    // sure that we do not miss the start time completely for very fast jobs
    // which are registered in the state JOB_QUEUE_RUNNING.
    if (new_status == JOB_QUEUE_WAITING || new_status == JOB_QUEUE_RUNNING)
        node->sim_start = time(NULL);

    if (!(new_status & JOB_QUEUE_COMPLETE_STATUS))
        return;
}

ERT_CLIB_SUBMODULE("queue", m) {
    using namespace py::literals;
    m.def("_refresh_status", [](Cwrap<job_queue_node_type> node,
                                Cwrap<queue_driver_type> driver) {
        pthread_mutex_lock(&node->data_mutex);
        job_status_type current_status = job_queue_node_get_status(node);

        if (!node->job_data) {
            pthread_mutex_unlock(&node->data_mutex);
            return std::make_pair<int, std::optional<std::string>>(
                int(current_status), std::nullopt);
        }

        std::optional<std::string> msg = std::nullopt;

        if ((current_status & JOB_QUEUE_RUNNING) &&
            (node->status_file && !(fs::exists(node->status_file)))) {
            // it's running, but not confirmed running.
            time_t runtime = time(nullptr) - node->sim_start;
            if (runtime >= MAX_CONFIRMED_WAIT) {
                std::string error_msg = fmt::format(
                    "max_confirm_wait ({}) has passed since sim_start"
                    "without success; {} is assumed dead (attempt {})",
                    MAX_CONFIRMED_WAIT, node->job_name, node->submit_attempt);
                logger->info(error_msg);
                msg = error_msg;
                job_status_type new_status = JOB_QUEUE_DO_KILL_NODE_FAILURE;
                job_queue_node_set_status(node, new_status);
            }
        }

        current_status = job_queue_node_get_status(node);
        if (current_status & JOB_QUEUE_CAN_UPDATE_STATUS) {
            job_status_type new_status =
                queue_driver_get_status(driver, node->job_data);
            if (new_status == JOB_QUEUE_EXIT)
                job_queue_node_fscanf_EXIT(node);
            job_queue_node_set_status(node, new_status);
            current_status = job_queue_node_get_status(node);
        }
        if (node->fail_message.has_value() and !msg.has_value())
            msg = node->fail_message;

        pthread_mutex_unlock(&node->data_mutex);
        return std::make_pair<int, std::optional<std::string>>(
            int(current_status), std::move(msg));
    });

    m.def("_submit", [](Cwrap<job_queue_node_type> node,
                        Cwrap<queue_driver_type> driver) {
        pthread_mutex_lock(&node->data_mutex);
        job_queue_node_set_status(node, JOB_QUEUE_SUBMITTED);
        void *job_data = queue_driver_submit_job(
            driver, node->run_cmd, node->num_cpu, node->run_path,
            node->job_name, node->argc, (const char **)node->argv);

        if (job_data == nullptr) {
            // In this case the status of the job itself will be
            // unmodified; i.e. it will still be WAITING, and a new attempt
            // to submit it will be performed in the next round.
            logger->warning("Failed to submit job {} (attempt {})",
                            node->job_name, node->submit_attempt);
            pthread_mutex_unlock(&node->data_mutex);
            return static_cast<int>(SUBMIT_DRIVER_FAIL);
        }

        logger->info("Submitted job {} (attempt {})", node->job_name,
                     node->submit_attempt);

        node->job_data = job_data;
        node->submit_attempt++;
        // The status JOB_QUEUE_SUBMITTED is internal, and not exported anywhere.
        // The job_queue_update_status() will update this to PENDING or RUNNING at
        // the next call. The important difference between SUBMITTED and WAITING is
        // that SUBMITTED have job_data != NULL and the job_queue_node free
        // function must be called on it.
        job_queue_node_set_status(node, JOB_QUEUE_SUBMITTED);
        pthread_mutex_unlock(&node->data_mutex);
        return static_cast<int>(SUBMIT_OK);
    });
    m.def("_kill",
          [](Cwrap<job_queue_node_type> node, Cwrap<queue_driver_type> driver) {
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
                  logger->warning("node_kill called but cannot kill {}",
                                  node->job_name);
              }
              pthread_mutex_unlock(&node->data_mutex);
              return result;
          });

    m.def("_get_submit_attempt",
          [](Cwrap<job_queue_node_type> node) { return node->submit_attempt; });
}
