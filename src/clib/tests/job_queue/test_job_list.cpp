#include "catch2/catch.hpp"
#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

TEST_CASE("job_list_allocate_list", "[job_list]") {
    auto *list = job_list_alloc();
    REQUIRE(job_list_get_size(list) == 0);
    job_list_free(list);
}

TEST_CASE("job_list_add_job", "[job_list]") {
    auto *list = job_list_alloc();
    auto *node = job_queue_node_alloc("name", "/tmp", "ls", 1, "", "");
    job_list_add_job(list, node);
    REQUIRE(job_list_get_size(list) == 1);
    REQUIRE(job_queue_node_get_queue_index(node) == 0);
    job_list_free(list);
}
