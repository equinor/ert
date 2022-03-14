/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'ui_return.h' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#ifndef ERT_UI_RETURN_H
#define ERT_UI_RETURN_H

#include <stdbool.h>

#include <ert/util/type_macros.hpp>
#include <ert/tooling.hpp>

typedef struct ui_return_struct ui_return_type;

typedef enum { UI_RETURN_OK = 1, UI_RETURN_FAIL = 2 } ui_return_status_enum;

extern "C" ui_return_type *ui_return_alloc(ui_return_status_enum status);
extern "C" void ui_return_free(ui_return_type *ui_return);
extern "C" ui_return_status_enum
ui_return_get_status(const ui_return_type *ui_return);
extern "C" int ui_return_get_error_count(const ui_return_type *ui_return);
extern "C" bool ui_return_add_error(ui_return_type *ui_return,
                                    const char *error_msg);
extern "C" void ui_return_add_help(ui_return_type *ui_return,
                                   const char *help_text);
extern "C" PY_USED const char *
ui_return_get_first_error(const ui_return_type *ui_return);
extern "C" const char *
ui_return_get_last_error(const ui_return_type *ui_return);
extern "C" PY_USED const char *
ui_return_get_help(const ui_return_type *ui_return);
extern "C" const char *ui_return_iget_error(const ui_return_type *ui_return,
                                            int index);


#endif
