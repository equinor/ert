#include <stdio.h>
#include <stdlib.h>

#include <ert/util/stringlist.hpp>
#include <ert/util/util.hpp>

#include <ert/config/config_error.hpp>

struct config_error_struct {
    stringlist_type *error_list;
};

config_error_type *config_error_alloc() {
    config_error_type *error = (config_error_type *)util_malloc(sizeof *error);
    error->error_list = stringlist_alloc_new();
    return error;
}

config_error_type *config_error_alloc_copy(const config_error_type *src_error) {
    config_error_type *config_error = config_error_alloc();
    stringlist_deep_copy(config_error->error_list, src_error->error_list);
    return config_error;
}

bool config_error_equal(const config_error_type *error1,
                        const config_error_type *error2) {
    return stringlist_equal(error1->error_list, error2->error_list);
}

void config_error_free(config_error_type *error) {
    stringlist_free(error->error_list);
    free(error);
}

void config_error_add(config_error_type *error, char *new_error) {
    stringlist_append_copy(error->error_list, new_error);
}

int config_error_count(const config_error_type *error) {
    return stringlist_get_size(error->error_list);
}

const char *config_error_iget(const config_error_type *error, int index) {
    return stringlist_iget(error->error_list, index);
}

void config_error_fprintf(const config_error_type *error, bool add_count,
                          FILE *stream) {
    int error_nr;

    for (error_nr = 0; error_nr < stringlist_get_size(error->error_list);
         error_nr++) {
        if (add_count)
            fprintf(stream, "  %02d: ", error_nr);

        fprintf(stream, "%s\n", stringlist_iget(error->error_list, error_nr));
    }
}
