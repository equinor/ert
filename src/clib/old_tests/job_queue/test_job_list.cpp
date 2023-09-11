#include <stdbool.h>
#include <stdlib.h>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

void test_create() {
    job_list_type *list = job_list_alloc();
    test_assert_int_equal(0, job_list_get_size(list));
    job_list_free(list);
}

void test_add_job() {
    job_list_type *list = job_list_alloc();
    job_queue_node_type *node = job_queue_node_alloc(
        "name", "/tmp", "ls", 0, stringlist_alloc_new(), 1, NULL, NULL);
    job_list_add_job(list, node);
    test_assert_int_equal(job_list_get_size(list), 1);
    test_assert_int_equal(job_queue_node_get_queue_index(node), 0);

    job_list_reset(list);
    test_assert_int_equal(0, job_list_get_size(list));
    job_list_free(list);
}

int main(int argc, char **argv) {
    test_create();
    test_add_job();
}
