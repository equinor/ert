#include <ert/util/test_util.hpp>
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>
#include <ert/util/util.h>

#include <ert/enkf/ert_workflow_list.hpp>

void test_create_workflow_list() {
    ert_workflow_list_type *wf_list = ert_workflow_list_alloc_empty(NULL);
    ert_workflow_list_free(wf_list);
}

void test_add_alias(const char *job) {
    ecl::util::TestArea ta("alias");
    ert_workflow_list_type *wf_list = ert_workflow_list_alloc_empty(NULL);
    ert_workflow_list_add_job(wf_list, "JOB", job);

    {
        FILE *stream = util_fopen("WF1", "w");
        fprintf(stream, "SCALE_STD 0.25\n");
        fclose(stream);
    }

    {
        FILE *stream = util_fopen("WF2", "w");
        fprintf(stream, "SCALE_STD 0.25\n");
        fclose(stream);
    }

    ert_workflow_list_add_workflow(wf_list, "WF1", "WF");
    test_assert_int_equal(1, ert_workflow_list_get_size(wf_list));
    test_assert_false(ert_workflow_list_has_workflow(wf_list, "WF1"));
    test_assert_true(ert_workflow_list_has_workflow(wf_list, "WF"));

    ert_workflow_list_add_alias(wf_list, "WF", "alias");
    test_assert_int_equal(2, ert_workflow_list_get_size(wf_list));
    test_assert_true(ert_workflow_list_has_workflow(wf_list, "WF"));
    test_assert_true(ert_workflow_list_has_workflow(wf_list, "alias"));
    test_assert_not_NULL(ert_workflow_list_get_workflow(wf_list, "WF"));
    test_assert_not_NULL(ert_workflow_list_get_workflow(wf_list, "alias"));

    ert_workflow_list_add_workflow(wf_list, "WF2", "WF");
    test_assert_int_equal(2, ert_workflow_list_get_size(wf_list));
    test_assert_true(ert_workflow_list_has_workflow(wf_list, "WF"));
    test_assert_true(ert_workflow_list_has_workflow(wf_list, "alias"));
    test_assert_not_NULL(ert_workflow_list_get_workflow(wf_list, "WF"));
    test_assert_not_NULL(ert_workflow_list_get_workflow(wf_list, "alias"));
}

int main(int argc, char **argv) {
    const char *job = argv[1];
    test_create_workflow_list();
    test_add_alias(job);
    exit(0);
}
