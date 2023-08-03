#include <stdlib.h>

#include <ert/job_queue/job_node.hpp>
#include <ert/util/test_util.hpp>

void test_create() {
    job_queue_node_type *node = job_queue_node_alloc(
        "name", "/tmp", "/bin/ls", 0, stringlist_alloc_new(), 1, NULL, NULL);
    job_queue_node_free(node);
}

void call_get_queue_index(void *arg) {
    auto node = static_cast<job_queue_node_type *>(arg);
    job_queue_node_get_queue_index(node);
}

void test_queue_index() {
    job_queue_node_type *node = job_queue_node_alloc(
        "name", "/tmp", "/bin/ls", 0, stringlist_alloc_new(), 1, NULL, NULL);
    test_assert_util_abort("job_queue_node_get_queue_index",
                           call_get_queue_index, node);
}

void test_path_does_not_exist() {
    job_queue_node_type *node =
        job_queue_node_alloc("name", "does-not-exist", "/bin/ls", 0,
                             stringlist_alloc_new(), 1, NULL, NULL);
    test_assert_NULL(node);
}

int main(int argc, char **argv) {
    util_install_signals();
    test_create();
    test_queue_index();
    test_path_does_not_exist();
}
