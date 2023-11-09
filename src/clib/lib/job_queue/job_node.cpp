#include <filesystem>
#include <fstream>
#include <string>

#include <cstdlib>
#include <pthread.h>
#include <tuple>

#include <ert/logging.hpp>

#include <ert/job_queue/job_node.hpp>
#include <ert/python.hpp>
#include <fmt/format.h>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("job_queue");

constexpr std::string_view exit_file = "ERROR";
constexpr std::string_view status_file = "STATUS";

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

static std::string tag_content(std::string xml, std::string tag) {
    std::string open_tag = fmt::format("<{}>", tag);
    std::string close_tag = fmt::format("</{}>", tag);

    auto start = xml.find(open_tag);
    auto end = xml.find(close_tag);

    if ((start != std::string::npos) && (end != std::string::npos)) {
        start += open_tag.length();
        return xml.substr(start, end - start);
    }
    return "";
}

static std::string add_tabs(std::string incoming) {
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
static void job_queue_node_fscanf_EXIT(job_queue_node_type *node) {
    auto exit_path = node->run_path / fs::path(exit_file);
    if (!fs::exists(exit_path)) {
        node->fail_message = fmt::format("EXIT file:{} not found", exit_path);
        return;
    }
    std::ifstream t(exit_path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string xml_str = buffer.str();

    std::string failed_job = tag_content(xml_str, "job");
    std::string error_reason = tag_content(xml_str, "reason");
    std::string stderr_file = tag_content(xml_str, "stderr_file");
    std::string stderr_capture = add_tabs(tag_content(xml_str, "stderr"));
    node->fail_message = fmt::format(
        "job {} failed with: '{}'\n\tstderr file: '{}',\n\tits contents:{}",
        failed_job, error_reason, stderr_file, stderr_capture);
}

int job_queue_node_get_queue_index(const job_queue_node_type *node) {
    return node->queue_index;
}

void job_queue_node_set_queue_index(job_queue_node_type *node,
                                    int queue_index) {
    node->queue_index = queue_index;
}

void job_queue_node_free(job_queue_node_type *node) { delete node; }

job_status_type job_queue_node_get_status(const job_queue_node_type *node) {
    return node->job_status;
}

job_queue_node_type *job_queue_node_alloc(const char *job_name,
                                          const char *run_path,
                                          const char *run_cmd, int num_cpu) {
    auto node = new job_queue_node_type;

    pthread_mutex_init(&node->data_mutex, nullptr);
    node->job_name = job_name;
    node->run_path = run_path;
    node->run_cmd = run_cmd;
    node->num_cpu = num_cpu;
    return node;
}

void job_queue_node_set_status(job_queue_node_type *node,
                               job_status_type new_status) {
    if (new_status == node->job_status)
        return;

    logger->debug("Set {}({}) to {}", node->job_name, node->queue_index,
                  job_status_names.at(new_status));
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
        // release the GIL
        py::gil_scoped_release release;

        pthread_mutex_lock(&node->data_mutex);
        job_status_type current_status = job_queue_node_get_status(node);

        if (!node->job_data) {
            pthread_mutex_unlock(&node->data_mutex);
            return std::make_pair<int, std::optional<std::string>>(
                int(current_status), std::nullopt);
        }

        std::optional<std::string> error_msg = std::nullopt;

        if (current_status & JOB_QUEUE_RUNNING && !node->confirmed_running) {
            node->confirmed_running =
                fs::exists(node->run_path / fs::path(status_file));

            if (!node->confirmed_running) {
                if ((time(nullptr) - node->sim_start) >= MAX_CONFIRMED_WAIT) {
                    error_msg = fmt::format(
                        "max_confirm_wait ({}) has passed since sim_start"
                        "without success; {} is assumed dead (attempt {})",
                        MAX_CONFIRMED_WAIT, node->job_name,
                        node->submit_attempt);
                    logger->info(error_msg.value());
                    job_queue_node_set_status(node,
                                              JOB_QUEUE_DO_KILL_NODE_FAILURE);
                    current_status = JOB_QUEUE_DO_KILL_NODE_FAILURE;
                }
            }
        }

        if (current_status & JOB_QUEUE_CAN_UPDATE_STATUS) {
            job_status_type new_status;
            try {
                new_status = queue_driver_get_status(driver, node->job_data);
            } catch (std::exception &err) {
                new_status = JOB_QUEUE_STATUS_FAILURE;
                error_msg = err.what();
            }

            if (new_status == JOB_QUEUE_EXIT)
                job_queue_node_fscanf_EXIT(node);

            job_queue_node_set_status(node, new_status);
            current_status = new_status;
        }

        if (node->fail_message.has_value() and !error_msg.has_value())
            error_msg = node->fail_message;

        pthread_mutex_unlock(&node->data_mutex);
        return std::make_pair(static_cast<int>(current_status), error_msg);
    });

    m.def("_submit", [](Cwrap<job_queue_node_type> node,
                        Cwrap<queue_driver_type> driver) {
        // release the GIL
        py::gil_scoped_release release;

        pthread_mutex_lock(&node->data_mutex);
        job_queue_node_set_status(node, JOB_QUEUE_SUBMITTED);
        void *job_data = nullptr;
        try {
            job_data = queue_driver_submit_job(
                driver, node->run_cmd.c_str(), node->num_cpu,
                node->run_path.c_str(), node->job_name.c_str());
        } catch (std::exception &err) {
            logger->warning("Failed to submit job {} (attempt {}) due to {}",
                            node->job_name, node->submit_attempt, err.what());
            pthread_mutex_unlock(&node->data_mutex);
            return static_cast<int>(SUBMIT_DRIVER_FAIL);
        }

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
              // release the GIL
              py::gil_scoped_release release;

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
