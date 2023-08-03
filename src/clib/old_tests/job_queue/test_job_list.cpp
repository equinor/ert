#include <stdbool.h>
#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

void test_create() {
    job_list_type *list = job_list_alloc();
    test_assert_int_equal(0, job_list_get_size(list));
    job_list_free(list);
}

void call_iget_job(void *arg) {
    auto job_list = static_cast<job_list_type *>(arg);
    job_list_iget_job(job_list, 10);
}

void test_add_job() {
    job_list_type *list = job_list_alloc();
    job_queue_node_type *node = job_queue_node_alloc(
        "name", "/tmp", "/bin/ls", 0, stringlist_alloc_new(), 1, NULL, NULL);
    job_list_add_job(list, node);
    test_assert_int_equal(job_list_get_size(list), 1);
    test_assert_int_equal(job_queue_node_get_queue_index(node), 0);
    test_assert_ptr_equal(node, job_list_iget_job(list, 0));

    {
        struct data_t {
            job_list_type *list;
            job_queue_node_type *node;
        } data{list, node};

        test_assert_util_abort(
            "job_queue_node_set_queue_index",
            [](void *data_) {
                auto data = reinterpret_cast<data_t *>(data_);
                job_list_add_job(data->list, data->node);
            },
            &data);
    }

    test_assert_util_abort("job_list_iget_job", call_iget_job, list);
    job_list_reset(list);
    test_assert_int_equal(0, job_list_get_size(list));
    job_list_free(list);
}

int main(int argc, char **argv) {
    util_install_signals();
    test_create();
    test_add_job();
}
