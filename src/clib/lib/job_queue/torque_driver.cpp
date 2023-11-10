#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <ert/abort.hpp>
#include <ert/job_queue/spawn.hpp>
#include <ert/job_queue/string_utils.hpp>
#include <ert/job_queue/torque_driver.hpp>
#include <ert/logging.hpp>
#include <ert/python.hpp>
#include <ert/util/util.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("job_queue.torque_driver");

namespace {
fs::path DEFAULT_QSTAT_CMD = TORQUE_DEFAULT_QSTAT_CMD;
}

struct torque_driver_struct {
    char *queue_name = nullptr;
    char *qsub_cmd = nullptr;
    char *qstat_cmd = nullptr;
    char *qstat_opts = nullptr;
    char *qdel_cmd = nullptr;
    char *num_cpus_per_node_char = nullptr;
    char *memory_per_job = nullptr;
    char *job_prefix = nullptr;
    char *num_nodes_char = nullptr;
    char *timeout_char = nullptr;
    bool keep_qsub_output = false;
    int num_cpus_per_node = 1;
    int num_nodes = 1;
    char *cluster_label = nullptr;
    int submit_sleep = 0;
    int timeout = 0;
};

struct torque_job_struct {
    long int torque_jobnr = 0;
    char *torque_jobnr_char = nullptr;
};

void *torque_driver_alloc() {
    auto torque_driver = new torque_driver_type;

    torque_driver_set_option(torque_driver, TORQUE_QSUB_CMD,
                             TORQUE_DEFAULT_QSUB_CMD);
    torque_driver_set_option(torque_driver, TORQUE_QSTAT_CMD,
                             DEFAULT_QSTAT_CMD.c_str());
    torque_driver_set_option(torque_driver, TORQUE_QSTAT_OPTIONS,
                             TORQUE_DEFAULT_QSTAT_OPTIONS);
    torque_driver_set_option(torque_driver, TORQUE_QDEL_CMD,
                             TORQUE_DEFAULT_QDEL_CMD);
    torque_driver_set_option(torque_driver, TORQUE_NUM_CPUS_PER_NODE, "1");
    torque_driver_set_option(torque_driver, TORQUE_NUM_NODES, "1");
    torque_driver_set_option(torque_driver, TORQUE_SUBMIT_SLEEP,
                             TORQUE_DEFAULT_SUBMIT_SLEEP);
    torque_driver_set_option(torque_driver, TORQUE_QUEUE_QUERY_TIMEOUT,
                             TORQUE_DEFAULT_QUEUE_QUERY_TIMEOUT);

    return torque_driver;
}

static void torque_driver_set_qsub_cmd(torque_driver_type *driver,
                                       const char *qsub_cmd) {
    driver->qsub_cmd = restrdup(driver->qsub_cmd, qsub_cmd);
}

static void torque_driver_set_qstat_cmd(torque_driver_type *driver,
                                        const char *qstat_cmd) {
    driver->qstat_cmd = restrdup(driver->qstat_cmd, qstat_cmd);
}

static void torque_driver_set_qstat_opts(torque_driver_type *driver,
                                         const char *qstat_opts) {
    driver->qstat_opts = restrdup(driver->qstat_opts, qstat_opts);
}

static void torque_driver_set_qdel_cmd(torque_driver_type *driver,
                                       const char *qdel_cmd) {
    driver->qdel_cmd = restrdup(driver->qdel_cmd, qdel_cmd);
}

static void torque_driver_set_queue_name(torque_driver_type *driver,
                                         const char *queue_name) {
    driver->queue_name = restrdup(driver->queue_name, queue_name);
}

static bool torque_driver_set_submit_sleep(torque_driver_type *driver,
                                           const char *submit_sleep) {
    double seconds_sleep;
    if (util_sscanf_double(submit_sleep, &seconds_sleep)) {
        driver->submit_sleep = (int)(seconds_sleep * 1000000);
        return true;
    }
    return false;
}

static bool torque_driver_set_num_nodes(torque_driver_type *driver,
                                        const char *num_nodes_char) {
    int num_nodes = 0;
    if (util_sscanf_int(num_nodes_char, &num_nodes)) {
        driver->num_nodes = num_nodes;
        driver->num_nodes_char =
            restrdup(driver->num_nodes_char, num_nodes_char);
        return true;
    }
    return false;
}

static bool
torque_driver_set_keep_qsub_output(torque_driver_type *driver,
                                   const char *keep_output_bool_as_char) {
    bool keep_output_parsed;

    if (util_sscanf_bool(keep_output_bool_as_char, &keep_output_parsed)) {
        driver->keep_qsub_output = keep_output_parsed;
        return true;
    }
    return false;
}

static void torque_driver_set_job_prefix(torque_driver_type *driver,
                                         const char *job_prefix) {
    driver->job_prefix = restrdup(driver->job_prefix, job_prefix);
}

static void torque_driver_set_cluster_label(torque_driver_type *driver,
                                            const char *cluster_label) {
    driver->cluster_label = restrdup(driver->cluster_label, cluster_label);
}

static bool
torque_driver_set_num_cpus_per_node(torque_driver_type *driver,
                                    const char *num_cpus_per_node_char) {
    int num_cpus_per_node = 0;
    if (util_sscanf_int(num_cpus_per_node_char, &num_cpus_per_node)) {
        driver->num_cpus_per_node = num_cpus_per_node;
        driver->num_cpus_per_node_char =
            restrdup(driver->num_cpus_per_node_char, num_cpus_per_node_char);
        return true;
    }
    return false;
}

static bool torque_driver_set_memory_per_job(torque_driver_type *driver,
                                             const char *memory_per_job) {
    driver->memory_per_job = restrdup(driver->memory_per_job, memory_per_job);
    return true;
}

static bool torque_driver_set_timeout(torque_driver_type *driver,
                                      const char *timeout_char) {
    int timeout = 0;
    if (util_sscanf_int(timeout_char, &timeout)) {
        driver->timeout = std::max(timeout, 0);
        driver->timeout_char = restrdup(driver->timeout_char, timeout_char);
        return true;
    }
    return false;
}

bool torque_driver_set_option(void *_driver, const char *option_key,
                              const void *value_) {
    const char *value = (const char *)value_;
    auto driver = static_cast<torque_driver_type *>(_driver);
    bool option_set = true;
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
        torque_driver_set_qsub_cmd(driver, value);
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
        torque_driver_set_qstat_cmd(driver, value);
    else if (strcmp(TORQUE_QSTAT_OPTIONS, option_key) == 0)
        torque_driver_set_qstat_opts(driver, value);
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
        torque_driver_set_qdel_cmd(driver, value);
    else if (strcmp(TORQUE_QUEUE, option_key) == 0)
        torque_driver_set_queue_name(driver, value);
    else if (strcmp(TORQUE_NUM_CPUS_PER_NODE, option_key) == 0)
        option_set = torque_driver_set_num_cpus_per_node(driver, value);
    else if (strcmp(TORQUE_MEMORY_PER_JOB, option_key) == 0)
        option_set = torque_driver_set_memory_per_job(driver, value);
    else if (strcmp(TORQUE_NUM_NODES, option_key) == 0)
        option_set = torque_driver_set_num_nodes(driver, value);
    else if (strcmp(TORQUE_KEEP_QSUB_OUTPUT, option_key) == 0)
        option_set = torque_driver_set_keep_qsub_output(driver, value);
    else if (strcmp(TORQUE_CLUSTER_LABEL, option_key) == 0)
        torque_driver_set_cluster_label(driver, value);
    else if (strcmp(TORQUE_JOB_PREFIX_KEY, option_key) == 0)
        torque_driver_set_job_prefix(driver, value);
    else if (strcmp(TORQUE_DEBUG_OUTPUT, option_key) == 0) {
        std::cerr << "DEBUG_OUTPUT queue option is deprecated, queue logging "
                     "can be found in jobqueue-log.txt\n";
        option_set = true;
    } else if (strcmp(TORQUE_SUBMIT_SLEEP, option_key) == 0)
        option_set = torque_driver_set_submit_sleep(driver, value);
    else if (strcmp(TORQUE_QUEUE_QUERY_TIMEOUT, option_key) == 0)
        option_set = torque_driver_set_timeout(driver, value);
    else
        option_set = false;
    return option_set;
}

const void *torque_driver_get_option(const void *_driver,
                                     const char *option_key) {
    const auto driver = static_cast<const torque_driver_type *>(_driver);
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
        return driver->qsub_cmd;
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
        return driver->qstat_cmd;
    else if (strcmp(TORQUE_QSTAT_OPTIONS, option_key) == 0)
        return driver->qstat_opts;
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
        return driver->qdel_cmd;
    else if (strcmp(TORQUE_QUEUE, option_key) == 0)
        return driver->queue_name;
    else if (strcmp(TORQUE_NUM_CPUS_PER_NODE, option_key) == 0)
        return driver->num_cpus_per_node_char;
    else if (strcmp(TORQUE_MEMORY_PER_JOB, option_key) == 0)
        return driver->memory_per_job;
    else if (strcmp(TORQUE_NUM_NODES, option_key) == 0)
        return driver->num_nodes_char;
    else if (strcmp(TORQUE_KEEP_QSUB_OUTPUT, option_key) == 0)
        return driver->keep_qsub_output ? "1" : "0";
    else if (strcmp(TORQUE_CLUSTER_LABEL, option_key) == 0)
        return driver->cluster_label;
    else if (strcmp(TORQUE_JOB_PREFIX_KEY, option_key) == 0)
        return driver->job_prefix;
    else if (strcmp(TORQUE_QUEUE_QUERY_TIMEOUT, option_key) == 0)
        return driver->timeout_char;
    else {
        throw std::runtime_error(
            fmt::format("option_id:{} not recognized for TORQUE driver",
                        std::string(option_key)));
        return nullptr;
    }
}

torque_job_type *torque_job_alloc() { return new torque_job_type; }

std::string build_resource_string(int num_nodes, std::string cluster_label,
                                  int num_cpus_per_node,
                                  std::string memory_per_job) {
    std::string resource_string = "nodes=" + std::to_string(num_nodes);

    if (!cluster_label.empty())
        resource_string.append(":" + cluster_label);

    resource_string.append(":ppn=" + std::to_string(num_cpus_per_node));

    if (!memory_per_job.empty())
        resource_string.append(":mem=" + memory_per_job);

    return resource_string;
}

#define TORQUE_ARGV_SIZE 12
static char **torque_driver_alloc_cmd(torque_driver_type *driver,
                                      const char *job_name,
                                      const char *submit_script) {

    char **argv =
        static_cast<char **>(calloc(TORQUE_ARGV_SIZE + 1, sizeof(char *)));
    CHECK_ALLOC(argv);
    int i = 0;

    argv[i++] = strdup(driver->qsub_cmd);

    if (driver->keep_qsub_output) {
        argv[i++] = strdup("-k");
        argv[i++] = strdup("oe");
    }

    argv[i++] = strdup("-l");
    std::string cluster_label{};
    if (driver->cluster_label != nullptr) {
        cluster_label = std::string(driver->cluster_label);
    }
    std::string memory_per_job{};
    if (driver->memory_per_job != nullptr) {
        memory_per_job = std::string(driver->memory_per_job);
    }
    argv[i++] =
        strdup(build_resource_string(driver->num_nodes, cluster_label,
                                     driver->num_cpus_per_node, memory_per_job)
                   .c_str());

    if (driver->queue_name != nullptr) {
        argv[i++] = strdup("-q");
        argv[i++] = strdup(driver->queue_name);
    }

    if (job_name != nullptr) {
        argv[i++] = strdup("-N");
        argv[i++] = strdup(job_name);
    }

    // Declare the job as not rerunnable
    argv[i++] = strdup("-r");
    argv[i++] = strdup("n");

    argv[i++] = strdup(submit_script);

    assert(i <= TORQUE_ARGV_SIZE);

    return argv;
}

static int torque_job_parse_qsub_stdout(const torque_driver_type *driver,
                                        const char *stdout_file,
                                        const char *stderr_file) {
    int jobid;
    FILE *stdout_stream = fopen(stdout_file, "r");
    if (!stdout_stream) {
        throw std::runtime_error("Unable to open qsub output: " +
                                 std::string(strerror(errno)));
    }

    char *jobid_string = util_fscanf_alloc_upto(stdout_stream, ".", false);

    bool possible_jobid;
    if (jobid_string == nullptr) {
        /* We get here if the '.' separator is not found */
        possible_jobid = util_fscanf_int(stdout_stream, &jobid);
        logger->debug("Torque job ID int: '{}'", jobid);
    } else {
        possible_jobid = util_sscanf_int(jobid_string, &jobid);
        logger->debug("Torque job ID string: '{}'", jobid);
    }

    if (!possible_jobid) {
        char *stdout_content =
            util_fread_alloc_file_content(stdout_file, nullptr);
        char *stderr_content =
            util_fread_alloc_file_content(stderr_file, nullptr);
        fprintf(stderr, "Failed to get torque job id from file: %s \n",
                stdout_file);
        fprintf(stderr, "qsub command: %s \n", driver->qsub_cmd);
        fprintf(stderr, "qsub output:  %s\n", stdout_content);
        fprintf(stderr, "qsub errors:  %s\n", stderr_content);
        free(stdout_content);
        free(stderr_content);
        jobid = -1;
    }
    free(jobid_string);
    fclose(stdout_stream);
    return jobid;
}

void torque_job_create_submit_script(const char *script_filename,
                                     const char *submit_cmd,
                                     const char *run_path) {
    if (submit_cmd == nullptr) {
        util_abort("%s: cannot create submit script, because there is no "
                   "executing commmand specified.",
                   __func__);
    }

    FILE *script_file = fopen(script_filename, "w");
    if (!script_file) {
        throw std::runtime_error("Unable to open submit script: " +
                                 std::string(strerror(errno)));
    }
    fprintf(script_file, "#!/bin/sh\n");

    fprintf(script_file, "%s", submit_cmd);
    fprintf(script_file, " %s", run_path);

    fclose(script_file);
}

static void torque_debug_spawn_status_info(torque_driver_type *driver,
                                           int status) {
    if (WIFEXITED((status))) {
        logger->debug("Torque spawn exited with status={}",
                      WEXITSTATUS((status)));
    } else if (WIFSIGNALED((status))) {
        logger->debug("Torque spawn killed by signal {}", WTERMSIG((status)));
    } else if (WIFSTOPPED((status))) {
        logger->debug("Torque spawn stopped by signal {}", WSTOPSIG((status)));
    } else if (WIFCONTINUED((status))) {
        logger->debug("Torque spawn continued");
    } else {
        logger->debug("Torque spawn failed with unknown status code: {}",
                      (status));
    }
}

static int torque_driver_submit_shell_job(torque_driver_type *driver,
                                          const char *run_path,
                                          const char *job_name,
                                          const char *submit_cmd, int num_cpu) {

    usleep(driver->submit_sleep);
    char *tmp_std_file =
        (char *)util_alloc_tmp_file("/tmp", "enkf-submit-std", true);
    char *tmp_err_file =
        (char *)util_alloc_tmp_file("/tmp", "enkf-submit-err", true);
    char *script_filename =
        (char *)util_alloc_filename(run_path, "qsub_script", "sh");

    logger->debug("Setting up submit stdout target '{}' for '{}'", tmp_std_file,
                  script_filename);
    logger->debug("Setting up submit stderr target '{}' for '{}'", tmp_err_file,
                  script_filename);
    torque_job_create_submit_script(script_filename, submit_cmd, run_path);
    int p_units_from_driver = driver->num_cpus_per_node * driver->num_nodes;
    if (num_cpu > p_units_from_driver) {
        throw std::runtime_error(fmt::format(
            "Error in config, job's config requires {} "
            "processing units, but config says {}: {}, and {}: "
            "{}, which multiplied becomes: {} \n",
            num_cpu, TORQUE_NUM_CPUS_PER_NODE, driver->num_cpus_per_node,
            TORQUE_NUM_NODES, driver->num_nodes, p_units_from_driver));
    }
    char **remote_argv =
        torque_driver_alloc_cmd(driver, job_name, script_filename);
    logger->debug("Submit arguments: {}", join_with_space(remote_argv));

    /* The qsub command might fail intermittently for acceptable reasons,
￼                  retry a couple of times with exponential sleep.  */
    int return_value = -1;
    int retry_interval = 2; /* seconds */
    int slept_time = 0;
    while (return_value != 0) {
        return_value = spawn_blocking(remote_argv, tmp_std_file, tmp_err_file);
        if (return_value != 0) {
            if (slept_time + retry_interval <= driver->timeout) {
                logger->debug("qsub failed for job {}, retrying in "
                              "{} seconds",
                              job_name, retry_interval);
                sleep(retry_interval);
                // Sleep some more at random, to avoid
                // synchronized retries from all threads:
                usleep(rand() % 2000000); // max 2 seconds
                slept_time += retry_interval;
                retry_interval *= 2;
            } else {
                logger->debug("qsub failed for job {}, no (more) retries",
                              job_name);
                break;
            }
        } else {
            if (slept_time > 0) {
                logger->debug("qsub succeeded for job {} after "
                              "waiting {} seconds",
                              job_name, slept_time);
            }
        }
    }
    if (return_value != 0) {
        torque_debug_spawn_status_info(driver, return_value);
    }
    for (int i = 0; i < TORQUE_ARGV_SIZE; i++) {
        free(remote_argv[i]);
    }
    free(remote_argv);

    int job_id =
        torque_job_parse_qsub_stdout(driver, tmp_std_file, tmp_err_file);

    util_unlink_existing(tmp_std_file);
    util_unlink_existing(tmp_err_file);
    free(tmp_std_file);
    free(tmp_err_file);

    return job_id;
}

void torque_job_free(torque_job_type *job) {

    free(job->torque_jobnr_char);
    delete job;
}

void torque_driver_free_job(void *_job) {

    auto job = static_cast<torque_job_type *>(_job);
    torque_job_free(job);
}

void *torque_driver_submit_job(void *_driver, std::string submit_cmd,
                               int num_cpu, fs::path run_path,
                               std::string job_name) {
    auto driver = static_cast<torque_driver_type *>(_driver);
    torque_job_type *job = torque_job_alloc();

    logger->debug("Submitting job in: {}", run_path);
    std::string local_job_name;
    if (driver->job_prefix)
        local_job_name = fmt::format("{}{}", driver->job_prefix, job_name);
    else
        local_job_name = job_name;

    job->torque_jobnr = torque_driver_submit_shell_job(
        driver, run_path.c_str(), local_job_name.c_str(), submit_cmd.c_str(),
        num_cpu);
    job->torque_jobnr_char = saprintf("%ld", job->torque_jobnr);

    logger->debug("Job:{} Id:{}", run_path, job->torque_jobnr);

    if (job->torque_jobnr > 0)
        return job;
    else {
        // The submit failed - the queue system shall handle
        // NULL return values.
        torque_job_free(job);
        return nullptr;
    }
}

/**
   Will return NULL if "something" fails; that again will be
   translated to JOB_QUEUE_STATUS_FAILURE - which the queue layer will
   just interpret as "No change in status". Possible failures are:

    1. The file capturing stdout is not created.
    2. Can not extract the correct status string from the stdout file.

*/
static job_status_type
torque_driver_get_qstat_status(torque_driver_type *driver,
                               const char *jobnr_char) {
    char *tmp_std_file =
        (char *)util_alloc_tmp_file("/tmp", "ert-qstat-std", true);
    char *tmp_err_file =
        (char *)util_alloc_tmp_file("/tmp", "ert-qstat-err", true);
    job_status_type status = JOB_QUEUE_STATUS_FAILURE;

    /* "qstat -f" means "full"/"long" output
     * (multiple lines of output pr. job)  */
    std::array argv{"-f", (const char *)driver->qstat_opts, jobnr_char};

    /* The qstat command might fail intermittently for acceptable reasons,
       retry a couple of times with exponential sleep. ERT pings qstat
       every second for every realization, thus the initial sleep time
       is 2 seconds. */
    bool qstat_succeeded = false;
    int retry_interval = 2; /* seconds */
    int slept_time = 0;
    while ((!qstat_succeeded) && (slept_time <= driver->timeout)) {
        int return_value =
            spawn_blocking(driver->qstat_cmd, argv.size(), argv.data(),
                           tmp_std_file, tmp_err_file);
        // A non-zero return value is trusted, but a zero return-value
        // is not trusted unless the output has nonzero length.
        // ERT never calls qstat unless it has already submitted something, and
        // can therefore assume that qstat results about Unknown Job Id are
        // failures (these have nonzero output length, but return value != 0)
        // that should trigger retries.
        if (std::error_code ec;
            fs::file_size(tmp_std_file, ec) > 0 && !ec && return_value == 0) {
            qstat_succeeded = true;
        }

        if (!qstat_succeeded) {
            if (slept_time + retry_interval <= driver->timeout) {
                logger->debug("qstat failed for job {} with exit code "
                              "{}, retrying in {} seconds",
                              jobnr_char, return_value, retry_interval);
                sleep(retry_interval);
                slept_time += retry_interval;
                retry_interval *= 2;
            } else {
                logger->debug("qstat failed for job %s, no (more) retries",
                              jobnr_char);
                break;
            }
        } else {
            if (slept_time > 0) {
                logger->debug("qstat succeeded for job {} after waiting "
                              "{} seconds",
                              jobnr_char, slept_time);
            }
        }
    }
    if (fs::exists(tmp_std_file)) {
        status = torque_driver_parse_status(tmp_std_file, jobnr_char);
        if (status != JOB_QUEUE_STATUS_FAILURE) {
            unlink(tmp_std_file);
            unlink(tmp_err_file);
        }
    } else
        fprintf(stderr,
                "No such file: %s - reading qstat status failed\n"
                "stderr: %s\n",
                tmp_std_file, tmp_err_file);
    free(tmp_std_file);
    free(tmp_err_file);

    return status;
}

job_status_type torque_driver_parse_status(const char *qstat_file,
                                           const char *jobnr_char) {
    job_status_type status = JOB_QUEUE_STATUS_FAILURE;

    int jobnr_no_namespace = -1;
    if (jobnr_char != nullptr) {
        /* Remove namespace from incoming job_id */
        std::string jobnr_namespaced(jobnr_char);
        int dot_position = jobnr_namespaced.find(".");
        if (dot_position != std::string::npos) {
            jobnr_namespaced.replace(dot_position, 1, " ");
        }
        std::stringstream(jobnr_namespaced) >> jobnr_no_namespace;
    }

    /* Parse the qstat output, looking only for requested job_id */
    std::string job_id_label("Job Id:");
    std::string job_state("_void_");
    int exit_status = 0;
    std::ifstream qstatoutput(qstat_file);
    qstatoutput.imbue(std::locale::classic());
    try {
        qstatoutput.exceptions(qstatoutput.failbit);
    } catch (const std::ios::failure &) {
        fprintf(stderr,
                "** Warning: Failed to parse job state for job %s "
                "from file '%s', file unreadable.\n",
                jobnr_char, qstat_file);
        return JOB_QUEUE_STATUS_FAILURE;
    }

    std::string line;
    int job_id_parser_state = -1;
    try {
        while (std::getline(qstatoutput, line)) {
            auto pos = line.find(job_id_label);
            if (pos != std::string::npos) {
                line.replace(0, job_id_label.length() + pos, "");

                // Remove namespace (Torque server name)
                int dot_position = line.find(".");
                if (dot_position != std::string::npos) {
                    line.replace(dot_position, 1, " ");
                }
                std::stringstream(line) >> job_id_parser_state;
            }

            if ((line.find("job_state") != std::string::npos) &&
                (job_id_parser_state == jobnr_no_namespace)) {
                std::string key, equalsign;
                try {
                    std::stringstream(line) >> key >> equalsign >> job_state;

                } catch (const std::ios::failure &) {
                    fprintf(stderr,
                            "** Warning: Failed to parse job state for job %s "
                            "from string '%s'.\n",
                            jobnr_char, line.c_str());
                }
            }

            if ((line.find("Exit_status") != std::string::npos) &&
                (job_id_parser_state == jobnr_no_namespace)) {
                std::string key, equalsign;
                try {
                    std::stringstream(line) >> key >> equalsign >> exit_status;

                } catch (const std::ios::failure &) {
                    fprintf(stderr,
                            "** Warning: Failed to parse exit status for job "
                            "%s from string '%s'.\n",
                            jobnr_char, line.c_str());
                }
            }
        }
    } catch (const std::ios::failure &) {
        // end-of-file
    }
    switch (job_state[0]) {
    case 'R':
        /* Job is running */
        status = JOB_QUEUE_RUNNING;
        break;
    case 'E':
        /* Job is exiting after having run */
        status = JOB_QUEUE_DONE;
        break;
    case 'F':
        /* PBS specific value: Job is finished */
        /* This is only returned in the alternative qstat format
               triggered with '-x' or '-H' option to qstat */
        status = JOB_QUEUE_DONE;
        break;
    case 'C':
        /* Job is completed after having run */
        status = JOB_QUEUE_DONE;
        break;
    case 'H':
        /* Job is held */
        status = JOB_QUEUE_PENDING;
        break;
    case 'Q':
        /* Job is queued, eligible to run or routed */
        status = JOB_QUEUE_PENDING;
        break;
    default:
        break;
    }

    if (exit_status != 0) {
        fprintf(stderr,
                "** Warning: Exit code %d from queue system on job: "
                "%s, job_state: %s\n",
                exit_status, jobnr_char, job_state.c_str());
        status = JOB_QUEUE_EXIT;
    }

    if (status == JOB_QUEUE_STATUS_FAILURE)
        fprintf(
            stderr,
            "** Warning: failed to get job status for job:%s from file:%s\n",
            jobnr_char, qstat_file);

    return status;
}

job_status_type torque_driver_get_job_status(void *_driver, void *_job) {
    auto driver = static_cast<torque_driver_type *>(_driver);
    auto job = static_cast<torque_job_type *>(_job);
    return torque_driver_get_qstat_status(driver, job->torque_jobnr_char);
}

void torque_driver_kill_job(void *_driver, void *_job) {

    char *tmp_std_file =
        (char *)util_alloc_tmp_file("/tmp", "ert-qdel-std", true);
    char *tmp_err_file =
        (char *)util_alloc_tmp_file("/tmp", "ert-qdel-err", true);

    auto driver = static_cast<torque_driver_type *>(_driver);
    auto job = static_cast<torque_job_type *>(_job);
    logger->debug("Killing Torque job: '{} {}'", driver->qdel_cmd,
                  job->torque_jobnr_char);

    /* The qdel command might fail intermittently for acceptable reasons,
           retry a couple of times with exponential sleep. */
    int return_value = -1;
    bool qdel_succeeded = false;
    int retry_interval = 2; /* seconds */
    int slept_time = 0;
    while ((return_value != 0) && (slept_time <= driver->timeout)) {
        return_value = spawn_blocking(driver->qdel_cmd, 1,
                                      (const char **)&job->torque_jobnr_char,
                                      tmp_std_file, tmp_err_file);
        if (return_value != 0) {
            if (slept_time + retry_interval <= driver->timeout) {
                logger->debug("qdel failed for job {} with exit code "
                              "{}, retrying in {} seconds",
                              job->torque_jobnr_char, return_value,
                              retry_interval);
                sleep(retry_interval);
                slept_time += retry_interval;
                retry_interval *= 2;
            } else {
                logger->debug("qdel failed for job {}, no (more) retries",
                              job->torque_jobnr_char);
                char *stderr_content =
                    util_fread_alloc_file_content(tmp_err_file, nullptr);
                logger->debug("qdel stderr: %s\n", stderr_content);
                free(stderr_content);
                break;
            }
        } else {
            if (slept_time > 0) {
                logger->debug("qdel succeeded for job {} after waiting "
                              "{} seconds",
                              job->torque_jobnr_char, slept_time);
            }
        }
    }
    free(tmp_std_file);
    free(tmp_err_file);
}

void torque_driver_free(torque_driver_type *driver) {
    free(driver->queue_name);
    free(driver->qsub_cmd);
    free(driver->qstat_cmd);
    free(driver->qstat_opts);
    free(driver->qdel_cmd);
    free(driver->num_cpus_per_node_char);
    free(driver->num_nodes_char);
    if (driver->job_prefix)
        free(driver->job_prefix);
    if (driver->cluster_label)
        free(driver->cluster_label);
    free(driver);
}

void torque_driver_free_(void *_driver) {
    auto driver = static_cast<torque_driver_type *>(_driver);
    torque_driver_free(driver);
}

ERT_CLIB_SUBMODULE("torque_driver", m) {
    using namespace py::literals;

    /* Determine installed 'qstat_proxy.sh' location */
    auto ert = py::module_::import("ert");

    py::str file = ert.attr("__file__");
    fs::path path = static_cast<std::string>(file);
    DEFAULT_QSTAT_CMD = path.parent_path() / "job_queue/qstat_proxy.sh";

    /* Expose for tests */
    auto pathlib = py::module_::import("pathlib");
    auto py_path = pathlib.attr("Path")(DEFAULT_QSTAT_CMD.c_str());
    m.add_object("DEFAULT_QSTAT_CMD", py_path);

    m.add_object("TORQUE_DRIVER_OPTIONS", py::cast(TORQUE_DRIVER_OPTIONS));

    py::enum_<job_status_type>(m, "JobStatusType", py::arithmetic())
        .export_values();

    m.def(
        "create_submit_script",
        [](const char *script_filename, const char *submit_cmd,
           const std::string &run_path) {
            torque_job_create_submit_script(script_filename, submit_cmd,
                                            run_path.c_str());
        },
        "script_filename"_a, "submit_cmd"_a, "job_argv"_a);

    m.def(
        "parse_status",
        [](const char *qstat_file, const char *jobnr_char) {
            return torque_driver_parse_status(qstat_file, jobnr_char);
        },
        "qstat_file"_a, "jobnr_char"_a);
    m.def(
        "build_resource_string",
        [](int num_nodes, const char *cluster_label, int num_cpus_per_node,
           const char *memory_per_job) {
            return build_resource_string(num_nodes, std::string(cluster_label),
                                         num_cpus_per_node,
                                         std::string(memory_per_job));
        },
        "num_nodes"_a, "cluster_label"_a, "num_cpus_per_node"_a,
        "memory_per_job"_a);
}
