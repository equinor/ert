#include "catch2/catch.hpp"
#include <cmath>
#include <cstdlib>
#include <ert/abort.hpp>
#include <pthread.h>

#include <ert/job_queue/job_queue_status.hpp>

TEST_CASE("job_node_allocate_status", "[job_status]") {
    auto *status = job_queue_status_alloc();
    REQUIRE(job_queue_status_get_count(status, JOB_QUEUE_DONE) == 0);
    REQUIRE(status != nullptr);
    job_queue_status_free(status);
}

void *add_sim(void *arg) {
    auto job_status = static_cast<job_queue_status_type *>(arg);
    job_queue_status_step(job_status, JOB_QUEUE_WAITING, 1);
    return nullptr;
}

void *user_exit(void *arg) {
    auto job_status = static_cast<job_queue_status_type *>(arg);
    job_queue_status_transition(job_status, JOB_QUEUE_WAITING,
                                JOB_QUEUE_DO_KILL);
    return nullptr;
}

void *user_done(void *arg) {
    auto job_status = static_cast<job_queue_status_type *>(arg);
    job_queue_status_transition(job_status, JOB_QUEUE_WAITING, JOB_QUEUE_DONE);
    return nullptr;
}

TEST_CASE("job_status_test_status_updates", "[job_status]") {
    int N = 15000, DN = 2 * N;
    pthread_t *thread_list = (pthread_t *)calloc(DN, sizeof *thread_list);
    CHECK_ALLOC(thread_list);

    int num_exit_threads = 0, num_done_threads = 0;
    auto *status = job_queue_status_alloc();

    REQUIRE(job_queue_status_transition(status, JOB_QUEUE_DONE,
                                        JOB_QUEUE_DONE) == 0); // no update

    // don't update on JOB_QUEUE_STATUS_FAILURE
    REQUIRE(job_queue_status_transition(status, JOB_QUEUE_DONE,
                                        JOB_QUEUE_STATUS_FAILURE) == 0);

    REQUIRE(job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL) == 0);

    for (int i = 0; i < DN; i++)
        add_sim(status);

    REQUIRE(DN == job_queue_status_get_count(status, JOB_QUEUE_WAITING));
    REQUIRE(DN == job_queue_status_get_count(status, JOB_QUEUE_WAITING +
                                                         JOB_QUEUE_RUNNING +
                                                         JOB_QUEUE_DONE));

    int idx = 0;
    while (true) {
        if ((idx % 2) == 0) {
            if (pthread_create(&thread_list[idx], NULL, user_exit, status) == 0)
                num_exit_threads++;
            else
                break;
        } else {
            if (pthread_create(&thread_list[idx], NULL, user_done, status) == 0)
                num_done_threads++;
            else
                break;
        }

        idx++;
        if (idx == N)
            break;
    }

    if ((num_done_threads + num_exit_threads) == 0) {
        fprintf(stderr, "No threads created\n");
        exit(1);
    }

    for (int i = 0; i < num_done_threads + num_exit_threads; i++)
        pthread_join(thread_list[i], NULL);

    REQUIRE(DN - num_done_threads - num_exit_threads ==
            job_queue_status_get_count(status, JOB_QUEUE_WAITING));

    REQUIRE(num_exit_threads ==
            job_queue_status_get_count(status, JOB_QUEUE_DO_KILL));
    REQUIRE(num_done_threads ==
            job_queue_status_get_count(status, JOB_QUEUE_DONE));

    REQUIRE(
        num_exit_threads + num_done_threads ==
        job_queue_status_get_count(status, JOB_QUEUE_DO_KILL + JOB_QUEUE_DONE));

    REQUIRE(DN == job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL));
    REQUIRE(DN == job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL - 1));
    job_queue_status_free(status);
    free(thread_list);
}

void add_and_count_job_status(job_queue_status_type *status_count,
                              job_status_type status_type, int count) {
    job_queue_status_step(status_count, status_type, 1);
    REQUIRE(job_queue_status_get_count(status_count, status_type) == 1);
    REQUIRE(job_queue_status_get_count(status_count, JOB_QUEUE_STATUS_ALL) ==
            count);
}

TEST_CASE("job_status_test_status_indices", "[job_status]") {
    job_queue_status_type *status = job_queue_status_alloc();
    int count = 1;
    add_and_count_job_status(status, JOB_QUEUE_NOT_ACTIVE, count++);
    add_and_count_job_status(status, JOB_QUEUE_WAITING, count++);
    add_and_count_job_status(status, JOB_QUEUE_SUBMITTED, count++);
    add_and_count_job_status(status, JOB_QUEUE_PENDING, count++);
    add_and_count_job_status(status, JOB_QUEUE_RUNNING, count++);
    add_and_count_job_status(status, JOB_QUEUE_DONE, count++);
    add_and_count_job_status(status, JOB_QUEUE_EXIT, count++);
    add_and_count_job_status(status, JOB_QUEUE_IS_KILLED, count++);
    add_and_count_job_status(status, JOB_QUEUE_DO_KILL, count++);
    add_and_count_job_status(status, JOB_QUEUE_SUCCESS, count++);
    add_and_count_job_status(status, JOB_QUEUE_STATUS_FAILURE, count++);
    add_and_count_job_status(status, JOB_QUEUE_FAILED, count++);
    add_and_count_job_status(status, JOB_QUEUE_DO_KILL_NODE_FAILURE, count++);
    job_queue_status_free(status);
}
