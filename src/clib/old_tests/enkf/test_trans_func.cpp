#include <ert/util/stringlist.h>
#include <ert/util/test_util.h>

#include <ert/enkf/trans_func.hpp>

void test_triangular() {
    stringlist_type *args = stringlist_alloc_new();
    stringlist_append_copy(args, "TRIANGULAR");
    stringlist_append_copy(args, "0");
    stringlist_append_copy(args, "0.5");
    stringlist_append_copy(args, "1.0");

    trans_func_type *trans_func = trans_func_alloc(args);
    test_assert_double_equal(trans_func_eval(trans_func, 0.0), 0.50);
    trans_func_free(trans_func);
    stringlist_free(args);
}

void test_triangular_assymetric() {
    stringlist_type *args = stringlist_alloc_new();
    stringlist_append_copy(args, "TRIANGULAR");
    stringlist_append_copy(args, "0");
    stringlist_append_copy(args, "1.0");
    stringlist_append_copy(args, "4.0");

    trans_func_type *trans_func = trans_func_alloc(args);
    test_assert_double_equal(trans_func_eval(trans_func, -1.0),
                             0.7966310411513150456286);
    test_assert_double_equal(trans_func_eval(trans_func, 1.1),
                             2.72407181575270778882286);
    trans_func_free(trans_func);
    stringlist_free(args);
}

void test_create() {
    {
        stringlist_type *args = stringlist_alloc_new();
        stringlist_append_copy(args, "UNKNOWN_FUNCTION");
        test_assert_NULL(trans_func_alloc(args));
        stringlist_free(args);
    }
    {
        stringlist_type *args = stringlist_alloc_new();
        stringlist_append_copy(args, "UNIFORM");
        stringlist_append_copy(args, "0");
        stringlist_append_copy(args, "1");

        trans_func_type *trans_func = trans_func_alloc(args);
        test_assert_double_equal(trans_func_eval(trans_func, 0.0), 0.50);
        trans_func_free(trans_func);

        stringlist_free(args);
    }
    {
        stringlist_type *args = stringlist_alloc_new();
        stringlist_append_copy(args, "UNIFORM");
        stringlist_append_copy(args, "0");
        stringlist_append_copy(args, "X");
        test_assert_NULL(trans_func_alloc(args));
        stringlist_free(args);
    }
    {
        stringlist_type *args = stringlist_alloc_new();
        stringlist_append_copy(args, "UNIFORM");
        stringlist_append_copy(args, "0");
        test_assert_NULL(trans_func_alloc(args));
        stringlist_free(args);
    }
}

int main(int argc, char **argv) {
    test_create();
    test_triangular();
    test_triangular_assymetric();
}
