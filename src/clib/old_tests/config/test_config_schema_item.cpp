#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/config/config_schema_item.hpp>

int main(int argc, char **argv) {
    config_schema_item_type *schema_item =
        config_schema_item_alloc("KW", false);

    test_assert_int_equal(config_schema_item_iget_type(schema_item, 1),
                          CONFIG_STRING);
    test_assert_int_equal(config_schema_item_iget_type(schema_item, 2),
                          CONFIG_STRING);

    config_schema_item_iset_type(schema_item, 0, CONFIG_INT);
    config_schema_item_iset_type(schema_item, 5, CONFIG_BOOL);

    test_assert_int_equal(config_schema_item_iget_type(schema_item, 0),
                          CONFIG_INT);
    test_assert_int_equal(config_schema_item_iget_type(schema_item, 1),
                          CONFIG_STRING);
    test_assert_int_equal(config_schema_item_iget_type(schema_item, 2),
                          CONFIG_STRING);
    test_assert_int_equal(config_schema_item_iget_type(schema_item, 5),
                          CONFIG_BOOL);

    config_schema_item_set_default_type(schema_item, CONFIG_FLOAT);
    test_assert_int_equal(config_schema_item_iget_type(schema_item, 7),
                          CONFIG_FLOAT);

    config_schema_item_free(schema_item);
    exit(0);
}
