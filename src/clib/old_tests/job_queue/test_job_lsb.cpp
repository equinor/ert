#include <dlfcn.h>
#include <stdbool.h>
#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/job_queue/lsb.hpp>

/*
  This test should ideally be run twice in two different environments;
  with and without dlopen() access to the lsf libraries.
*/

int main(int argc, char **argv) {
    lsb_type *lsb = lsb_alloc();

    test_assert_not_NULL(lsb);
    if (!lsb_ready(lsb)) {
        const stringlist_type *error_list = lsb_get_error_list(lsb);
        stringlist_fprintf(error_list, "\n", stdout);
    }

    if (dlopen("libbat.so", RTLD_NOW | RTLD_GLOBAL))
        test_assert_true(lsb_ready(lsb));
    else
        test_assert_false(lsb_ready(lsb));

    lsb_free(lsb);
    exit(0);
}
