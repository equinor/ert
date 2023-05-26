#ifndef ERT_CONFIG_CONTENT_ITEM_H
#define ERT_CONFIG_CONTENT_ITEM_H

#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>

#include <ert/config/config_content_node.hpp>
#include <ert/config/config_path_elm.hpp>
#include <ert/config/config_schema_item.hpp>

typedef struct config_content_item_struct config_content_item_type;

extern "C" int
config_content_item_get_size(const config_content_item_type *item);
config_content_node_type *
config_content_item_get_last_node(const config_content_item_type *item);
extern "C" config_content_node_type *
config_content_item_iget_node(const config_content_item_type *item, int index);
const char *config_content_item_iget(const config_content_item_type *item,
                                     int occurence, int index);
void config_content_item_clear(config_content_item_type *item);
extern "C" void config_content_item_free(config_content_item_type *item);
void config_content_item_free__(void *arg);
extern "C" config_content_item_type *
config_content_item_alloc(const config_schema_item_type *schema,
                          const config_path_elm_type *path_elm);
config_content_node_type *
config_content_item_alloc_node(const config_content_item_type *item,
                               const config_path_elm_type *path_elm);
const config_schema_item_type *
config_content_item_get_schema(const config_content_item_type *item);
const config_path_elm_type *
config_content_item_get_path_elm(const config_content_item_type *item);

#endif
