#include <stdbool.h>
#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/job_queue/lsb.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/lsf_job_stat.hpp>

void test_server(lsf_driver_type *driver, const char *server,
                 lsf_submit_method_enum submit_method) {
    lsf_driver_set_option(driver, LSF_SERVER, server);
    test_assert_true(lsf_driver_get_submit_method(driver) == submit_method);
}

/*
  This test should ideally be run twice in two different environments;
  with and without dlopen() access to the lsf libraries.
*/

int main(int argc, char **argv) {
    lsf_submit_method_enum submit_NULL;
    lsb_type *lsb = lsb_alloc();
    if (lsb_ready(lsb))
        submit_NULL = LSF_SUBMIT_INTERNAL;
    else
        submit_NULL = LSF_SUBMIT_LOCAL_SHELL;

    test_server(driver, NULL, submit_NULL);
    test_server(driver, "LoCaL", LSF_SUBMIT_LOCAL_SHELL);
    test_server(driver, "LOCAL", LSF_SUBMIT_LOCAL_SHELL);
    test_server(driver, "XLOCAL", LSF_SUBMIT_REMOTE_SHELL);
    test_server(driver, NULL, submit_NULL);
    test_server(driver, "NULL", submit_NULL);
    test_server(driver, "be-grid01", LSF_SUBMIT_REMOTE_SHELL);
    printf("Servers OK\n");

    lsb_free(lsb);
    exit(0);
}
