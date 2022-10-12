#include <stdlib.h>

#include <ert/util/vector.hpp>

#include <ert/config/config_path_elm.hpp>
#include <ert/config/config_path_stack.hpp>

#define CONFIG_PATH_STACK_TYPE_ID 86751520

struct config_path_stack_struct {
    vector_type *storage;
    vector_type *stack;
};

config_path_stack_type *config_path_stack_alloc() {
    config_path_stack_type *path_stack =
        (config_path_stack_type *)util_malloc(sizeof *path_stack);
    path_stack->storage = vector_alloc_new();
    path_stack->stack = vector_alloc_new();
    return path_stack;
}

void config_path_stack_free(config_path_stack_type *path_stack) {
    vector_free(path_stack->storage);
    vector_free(path_stack->stack);
}

void config_path_stack_append(config_path_stack_type *path_stack,
                              config_path_elm_type *path_elm) {
    vector_append_owned_ref(path_stack->storage, path_elm,
                            config_path_elm_free__);
    vector_append_ref(path_stack->stack, path_elm);
}

int config_path_stack_size(const config_path_stack_type *path_stack) {
    return vector_get_size(path_stack->stack);
}

const config_path_elm_type *
config_path_stack_get_last(const config_path_stack_type *path_stack) {
    return (const config_path_elm_type *)vector_get_last_const(
        path_stack->stack);
}

void config_path_stack_pop(config_path_stack_type *path_stack) {
    vector_pop_back(path_stack->stack);
}
