/*
   Copyright (C) 2015  Equinor ASA, Norway.

   The file 'job_node_test.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/
#include <stdlib.h>
#include <stdbool.h>

#include <ert/util/test_util.hpp>

#include <ert/job_queue/job_node.hpp>
#include <ert/job_queue/job_list.hpp>

void test_create() {
    job_list_type *list = job_list_alloc();
    test_assert_true(job_list_is_instance(list));
    test_assert_int_equal(0, job_list_get_size(list));
    job_list_free(list);
}

void call_iget_job(void *arg) {
    job_list_type *job_list = job_list_safe_cast(arg);
    job_list_iget_job(job_list, 10);
}

void test_add_job() {
    job_list_type *list = job_list_alloc();
    job_queue_node_type *node =
        job_queue_node_alloc_simple("name", "/tmp", "/bin/ls", 0, NULL);
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
