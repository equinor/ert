#include <ert/res_util/subst_list.hpp>
#include <ert/util/test_util.hpp>
#include <stdarg.h>
#include <stdexcept>
#include <stdlib.h>

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

void expect_exception(const char *arg_string, const char *err_msg) {
    subst_list_type *subst_list = subst_list_alloc();
    bool success = false;

    try {
        subst_list_add_from_string(subst_list, arg_string);
    } catch (const std::invalid_argument &err) {
        if (strcmp(err.what(), err_msg) == 0)
            success = true;
    }

    test_assert_true(success);
    subst_list_free(subst_list);
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
    expect_exception(",", "Missing '=' in argument");
    expect_exception("x", "Missing '=' in argument");
    expect_exception(", x=1", "Missing '=' in argument");
    expect_exception("x=1,,y=2", "Missing '=' in argument");
    expect_exception("x=1,", "Trailing comma in argument list");
    expect_exception("x=", "Missing value in argument list");
    expect_exception("=x", "Missing key in argument list");
    expect_exception("x='a", "Missing string delimiter in argument");
    expect_exception("x=a'", "Missing string delimiter in argument");
    expect_exception("'x'=1", "Key cannot contain quotation marks");
    expect_exception("\"x\"=1", "Key cannot contain quotation marks");
}
