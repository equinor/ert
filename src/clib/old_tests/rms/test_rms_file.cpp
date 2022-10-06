#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/rms/rms_file.hpp>
#include <ert/rms/rms_util.hpp>

void test_rms_file_fread_alloc_data_tag(rms_file_type *rms_file) {
    rms_tag_type *parameter_tag =
        rms_file_fread_alloc_tag(rms_file, "parameter", NULL, NULL);
    test_assert_not_NULL(parameter_tag);
    test_assert_string_equal("parameter", rms_tag_get_name(parameter_tag));
    rms_tag_free(parameter_tag);
}

void test_rms_file_fread_alloc_data_tagkey(rms_file_type *rms_file) {
    rms_tagkey_type *name_tagkey =
        rms_file_fread_alloc_data_tagkey(rms_file, "parameter", NULL, NULL);
    test_assert_not_NULL(name_tagkey);
    test_assert_int_equal(rms_float_type, rms_tagkey_get_rms_type(name_tagkey));
    rms_tagkey_free(name_tagkey);
}

int main(int argc, char **argv) {
    const char *filename = argv[1];
    rms_file_type *rms_file = rms_file_alloc(filename, false);
    test_assert_not_NULL(rms_file);

    test_rms_file_fread_alloc_data_tag(rms_file);
    test_rms_file_fread_alloc_data_tagkey(rms_file);

    rms_file_free(rms_file);
    exit(0);
}
