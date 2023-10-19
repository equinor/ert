#include "catch2/catch.hpp"
#include <ert/job_queue/job_node.hpp>

TEST_CASE("job_node_test_allocate_node", "[job_node]") {
    auto *node = job_queue_node_alloc("name", "/tmp", "ls", 1, NULL, NULL);
    REQUIRE(node != nullptr);
    job_queue_node_free(node);
}

TEST_CASE("job_node_test_allocate_node_when_invalid_path", "[job_node]") {
    auto *node =
        job_queue_node_alloc("name", "does-not-exist", "ls", 1, NULL, NULL);
    REQUIRE_FALSE(node != nullptr);
}
