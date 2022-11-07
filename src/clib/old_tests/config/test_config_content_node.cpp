#include <filesystem>
#include <stdlib.h>

#include <ert/util/hash.hpp>
#include <ert/util/test_util.hpp>

#include <ert/config/config_content_node.hpp>

int main(int argc, char **argv) {
    config_schema_item_type *schema = config_schema_item_alloc("TEST", true);
    config_path_elm_type *cwd =
        config_path_elm_alloc(std::filesystem::current_path(), NULL);
    {
        config_content_node_type *node = config_content_node_alloc(schema, cwd);
        config_content_node_add_value(node, "KEY1:VALUE1");
        config_content_node_add_value(node, "KEY2:VALUE2");
        config_content_node_add_value(node, "KEY3:VALUE3");
        config_content_node_add_value(node, "KEYVALUE");

        test_assert_int_equal(config_content_node_get_size(node), 4);
        test_assert_string_equal(config_content_node_iget(node, 0),
                                 "KEY1:VALUE1");
        test_assert_string_equal(config_content_node_iget(node, 2),
                                 "KEY3:VALUE3");

        test_assert_string_equal(
            config_content_node_get_full_string(node, ","),
            "KEY1:VALUE1,KEY2:VALUE2,KEY3:VALUE3,KEYVALUE");

        config_content_node_free(node);
    }
    config_path_elm_free(cwd);
    config_schema_item_free(schema);
    exit(0);
}
