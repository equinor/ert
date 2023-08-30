#include <cmath>
#include <pthread.h>
#include <stdlib.h>

#include <ert/job_queue/job_queue_status.hpp>
#include <ert/util/test_util.hpp>

void call_get_status(void *arg) {
    auto job_status = static_cast<job_queue_status_type *>(arg);
    job_queue_status_get_count(
        job_status,
        pow(2,
            JOB_QUEUE_MAX_STATE)); // This enum value is completly missing; should give util_abort.
}

void test_create() {
    job_queue_status_type *status = job_queue_status_alloc();
    test_assert_int_equal(job_queue_status_get_count(status, JOB_QUEUE_DONE),
                          0);
    test_assert_util_abort("job_queue_status_get_count", call_get_status,
                           status);
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

void test_update() {
    int N = 15000;
    pthread_t *thread_list = (pthread_t *)calloc(2 * N, sizeof *thread_list);

    int num_exit_threads = 0, num_done_threads = 0;
    job_queue_status_type *status = job_queue_status_alloc();

    test_assert_int_equal(
        0, job_queue_status_transition(status, JOB_QUEUE_DONE,
                                       JOB_QUEUE_DONE)); // no update
    // don't update on JOB_QUEUE_STATUS_FAILURE
    test_assert_int_equal(
        0, job_queue_status_transition(status, JOB_QUEUE_DONE,
                                       JOB_QUEUE_STATUS_FAILURE));

    int total_count = job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL);
    test_assert_int_equal(0, total_count);

    for (int i = 0; i < 2 * N; i++)
        add_sim(status);

    test_assert_int_equal(
        2 * N, job_queue_status_get_count(status, JOB_QUEUE_WAITING));
    test_assert_int_equal(
        2 * N,
        job_queue_status_get_count(
            status, JOB_QUEUE_WAITING + JOB_QUEUE_RUNNING + JOB_QUEUE_DONE));

    int idx = 0;
    while (true) {
        int thread_status;

        if ((idx % 2) == 0) {
            thread_status =
                pthread_create(&thread_list[idx], NULL, user_exit, status);
            if (thread_status == 0)
                num_exit_threads++;
            else
                break;
        } else {
            thread_status =
                pthread_create(&thread_list[idx], NULL, user_done, status);
            if (thread_status == 0)
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

    test_assert_int_equal(
        2 * N - num_done_threads - num_exit_threads,
        job_queue_status_get_count(status, JOB_QUEUE_WAITING));
    test_assert_int_equal(num_exit_threads, job_queue_status_get_count(
                                                status, JOB_QUEUE_DO_KILL));
    test_assert_int_equal(num_done_threads,
                          job_queue_status_get_count(status, JOB_QUEUE_DONE));

    test_assert_int_equal(
        num_exit_threads + num_done_threads,
        job_queue_status_get_count(status, JOB_QUEUE_DO_KILL + JOB_QUEUE_DONE));

    total_count = job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL);
    int total_count_ex_unknown =
        job_queue_status_get_count(status, JOB_QUEUE_STATUS_ALL - 1);

    test_assert_int_equal(2 * N, total_count);
    test_assert_int_equal(2 * N, total_count_ex_unknown);
    job_queue_status_free(status);
    free(thread_list);
}

/*
  The job_queue_status_step( ) and the job_queue_status_get_count( )
  functions use two different and independent implementations
  internally; that is the reason for this seemingly quite trivial and
  not-very-interesting test.
*/

void add_and_count_job_status(job_queue_status_type *status_count,
                              job_status_type status_type, int count) {
    job_queue_status_step(status_count, status_type, 1);

    test_assert_int_equal(job_queue_status_get_count(status_count, status_type),
                          1);
    test_assert_int_equal(
        job_queue_status_get_count(status_count, JOB_QUEUE_STATUS_ALL), count);
}

void test_index() {
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
    add_and_count_job_status(status, JOB_QUEUE_RUNNING_DONE_CALLBACK, count++);
    add_and_count_job_status(status, JOB_QUEUE_RUNNING_EXIT_CALLBACK, count++);
    add_and_count_job_status(status, JOB_QUEUE_STATUS_FAILURE, count++);
    add_and_count_job_status(status, JOB_QUEUE_FAILED, count++);
    add_and_count_job_status(status, JOB_QUEUE_DO_KILL_NODE_FAILURE, count++);
    job_queue_status_free(status);
}

int main(int argc, char **argv) {
    util_install_signals();
    test_create();
    test_index();
    test_update();
}
