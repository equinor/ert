#ifndef ERT_CONFIG_PATH_STACK_H
#define ERT_CONFIG_PATH_STACK_H

#include <ert/config/config_path_elm.hpp>

typedef struct config_path_stack_struct config_path_stack_type;

void config_path_stack_free(config_path_stack_type *path_stack);
config_path_stack_type *config_path_stack_alloc();
void config_path_stack_append(config_path_stack_type *path_stack,
                              config_path_elm_type *path_elm);
int config_path_stack_size(const config_path_stack_type *path_stack);
const config_path_elm_type *
config_path_stack_get_last(const config_path_stack_type *path_stack);
void config_path_stack_pop(config_path_stack_type *path_stack);

#endif
