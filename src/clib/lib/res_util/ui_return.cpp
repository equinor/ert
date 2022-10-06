#include <stdlib.h>
#include <string.h>

#include <ert/util/stringlist.hpp>

#include <ert/res_util/ui_return.hpp>

#define UI_RETURN_TYPE_ID 6122209

struct ui_return_struct {
    UTIL_TYPE_ID_DECLARATION;
    ui_return_status_enum status;
    stringlist_type *error_list;
    char *help_text;
};

UTIL_IS_INSTANCE_FUNCTION(ui_return, UI_RETURN_TYPE_ID)

ui_return_type *ui_return_alloc(ui_return_status_enum status) {
    ui_return_type *ui_return =
        (ui_return_type *)util_malloc(sizeof *ui_return);
    UTIL_TYPE_ID_INIT(ui_return, UI_RETURN_TYPE_ID);
    ui_return->status = status;
    ui_return->help_text = NULL;
    ui_return->error_list = stringlist_alloc_new();
    return ui_return;
}

void ui_return_free(ui_return_type *ui_return) {
    stringlist_free(ui_return->error_list);
    free(ui_return->help_text);
    free(ui_return);
}

ui_return_status_enum ui_return_get_status(const ui_return_type *ui_return) {
    return ui_return->status;
}

bool ui_return_add_error(ui_return_type *ui_return, const char *error_msg) {
    if (ui_return->status != UI_RETURN_OK)
        stringlist_append_copy(ui_return->error_list, error_msg);

    return (ui_return->status != UI_RETURN_OK);
}

int ui_return_get_error_count(const ui_return_type *ui_return) {
    return stringlist_get_size(ui_return->error_list);
}

const char *ui_return_get_first_error(const ui_return_type *ui_return) {
    if (stringlist_get_size(ui_return->error_list))
        return stringlist_front(ui_return->error_list);
    else
        return NULL;
}

const char *ui_return_get_last_error(const ui_return_type *ui_return) {
    if (stringlist_get_size(ui_return->error_list))
        return stringlist_back(ui_return->error_list);
    else
        return NULL;
}

const char *ui_return_iget_error(const ui_return_type *ui_return, int index) {
    return stringlist_iget(ui_return->error_list, index);
}

const char *ui_return_get_help(const ui_return_type *ui_return) {
    return ui_return->help_text;
}

void ui_return_add_help(ui_return_type *ui_return, const char *help_text) {
    if (ui_return->help_text) {
        int new_length =
            strlen(ui_return->help_text) + strlen(help_text) + 1 + 1;
        ui_return->help_text = (char *)util_realloc(
            ui_return->help_text, new_length * sizeof *ui_return->help_text);

        strcat(ui_return->help_text, " ");
        strcat(ui_return->help_text, help_text);
    } else
        ui_return->help_text = util_alloc_string_copy(help_text);
}
