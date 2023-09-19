#include <cassert>
#include <stdlib.h>

#include <ert/job_queue/job_node.hpp>

void test_create() {
    job_queue_node_type *node = job_queue_node_alloc(
        "name", "/tmp", "ls", 0, stringlist_alloc_new(), 1, NULL, NULL);
    job_queue_node_free(node);
}

void test_path_does_not_exist() {
    job_queue_node_type *node =
        job_queue_node_alloc("name", "does-not-exist", "ls", 0,
                             stringlist_alloc_new(), 1, NULL, NULL);
    assert(node == nullptr);
}

int main(int argc, char **argv) {
    test_create();
    test_path_does_not_exist();
}
