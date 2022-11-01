#include <ert/res_util/subst_list.hpp>
#include <ert/util/test_util.hpp>
#include <stdarg.h>
#include <stdlib.h>

void call_func_util_abort(void *args) {
    subst_list_type *subst_list = subst_list_alloc();
    subst_list_add_from_string(subst_list, (char *)args);
    subst_list_free(subst_list);
}

void test_valid_arg_string(const char *arg_string, int arg_count, ...) {
    va_list ap;
    subst_list_type *subst_list = subst_list_alloc();
    subst_list_add_from_string(subst_list, arg_string);
    test_assert_int_equal(subst_list_get_size(subst_list), arg_count);
    va_start(ap, arg_count);
    for (int i = 0; i < arg_count; i++) {
        test_assert_string_equal(subst_list_iget_key(subst_list, i),
                                 va_arg(ap, const char *));
        test_assert_string_equal(subst_list_iget_value(subst_list, i),
                                 va_arg(ap, const char *));
    }
    va_end(ap);
    subst_list_free(subst_list);
}

void test_invalid_arg_string(const char *arg_string) {
    char *tmp = util_alloc_substring_copy(arg_string, 0, strlen(arg_string));
    test_assert_util_abort("subst_list_add_from_string", call_func_util_abort,
                           tmp);
    free(tmp);
}

int main(int argc, char **argv) {
    test_valid_arg_string("", 0);
    test_valid_arg_string(" ", 0);
    test_valid_arg_string("\t", 0);
    test_valid_arg_string("x=1", 1, "x", "1");
    test_valid_arg_string(" x=1", 1, "x", "1");
    test_valid_arg_string("x=1 ", 1, "x", "1");
    test_valid_arg_string("x = 1", 1, "x", "1");
    test_valid_arg_string("x=1,y=2", 2, "x", "1", "y", "2");
    test_valid_arg_string("x=1, y=2", 2, "x", "1", "y", "2");
    test_valid_arg_string("x='a'", 1, "x", "'a'");
    test_valid_arg_string("x='a',y="
                          "\"a,b\"",
                          2, "x", "'a'", "y", "\"a,b\"");
    test_valid_arg_string("x='a' 'b'", 1, "x", "'a' 'b'");

    test_invalid_arg_string(",");
    test_invalid_arg_string(", x=1");
    test_invalid_arg_string("x=1,");
    test_invalid_arg_string("x=1,,y=2");
    test_invalid_arg_string("x");
    test_invalid_arg_string("x=");
    test_invalid_arg_string("=x");
    test_invalid_arg_string("x='a");
    test_invalid_arg_string("x=a'");
    test_invalid_arg_string("'x'=1");
    test_invalid_arg_string("\"x\"=1");
}
