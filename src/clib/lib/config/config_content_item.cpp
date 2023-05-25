#include <ert/util/stringlist.hpp>
#include <ert/util/vector.hpp>

#include <ert/config/config_content_item.hpp>
#include <ert/config/config_content_node.hpp>
#include <ert/config/config_schema_item.hpp>

struct config_content_item_struct {
    const config_schema_item_type *schema;
    vector_type *nodes;
    const config_path_elm_type *path_elm;
};

/**
   This function counts the number of times a config item has been
   set. Referring again to the example at the top:

     config_content_item_get_occurences( "KEY1" )

   will return 2.
*/
int config_content_item_get_size(const config_content_item_type *item) {
    return vector_get_size(item->nodes);
}

config_content_node_type *
config_content_item_get_last_node(const config_content_item_type *item) {
    return (config_content_node_type *)vector_get_last(item->nodes);
}

config_content_node_type *
config_content_item_iget_node(const config_content_item_type *item, int index) {
    return (config_content_node_type *)vector_iget(item->nodes, index);
}

const char *config_content_item_iget(const config_content_item_type *item,
                                     int occurence, int index) {
    const config_content_node_type *node =
        config_content_item_iget_node(item, occurence);
    const stringlist_type *src_list = config_content_node_get_stringlist(node);
    return stringlist_iget(src_list, index);
}

/**
   Used to reset an item is the special string 'CLEAR_STRING'
   is found as the only argument:

   OPTION V1
   OPTION V2 V3 V4
   OPTION __RESET__
   OPTION V6

   In this case OPTION will get the value 'V6'. The example given
   above is a bit contrived; this option is designed for situations
   where several config files are parsed serially; and the user can
   not/will not update the first.
*/
void config_content_item_clear(config_content_item_type *item) {
    vector_clear(item->nodes);
}

void config_content_item_free(config_content_item_type *item) {
    vector_free(item->nodes);
    free(item);
}

void config_content_item_free__(void *arg) {
    auto content_item = static_cast<config_content_item_type *>(arg);
    config_content_item_free(content_item);
}

config_content_item_type *
config_content_item_alloc(const config_schema_item_type *schema,
                          const config_path_elm_type *path_elm) {
    config_content_item_type *content_item =
        (config_content_item_type *)util_malloc(sizeof *content_item);
    content_item->schema = schema;
    content_item->nodes = vector_alloc_new();
    content_item->path_elm = path_elm;
    return content_item;
}

config_content_node_type *
config_content_item_alloc_node(const config_content_item_type *item,
                               const config_path_elm_type *path_elm) {
    config_content_node_type *node =
        config_content_node_alloc(item->schema, path_elm);
    vector_append_owned_ref(item->nodes, node, config_content_node_free__);
    return node;
}

const config_schema_item_type *
config_content_item_get_schema(const config_content_item_type *item) {
    return item->schema;
}

const config_path_elm_type *
config_content_item_get_path_elm(const config_content_item_type *item) {
    return item->path_elm;
}
