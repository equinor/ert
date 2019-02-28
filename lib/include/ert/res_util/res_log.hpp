/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'res_log.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef RESLOG_H
#define RESLOG_H

#include <stdio.h>
#include <stdbool.h>

#include <ert/res_util/log.hpp>

#ifdef __cplusplus
extern "C" {
#endif


bool res_log_init_log(message_level_type log_level,const char * log_file_name, bool verbose);
bool res_log_init_log_default_log_level(const char * log_file_name, bool verbose);
void res_log_close();
const char * res_log_get_filename();

void res_log_add_message(message_level_type message_level, const char* message);

void res_log_debug(const char* msg);
void res_log_info(const char* msg);
void res_log_warning(const char* msg);
void res_log_error(const char* msg);
void res_log_critical(const char* msg);

void res_log_fdebug(const char * fmt, ...);
void res_log_finfo(const char * fmt, ...);
void res_log_fwarning(const char * fmt, ...);
void res_log_ferror(const char * fmt, ...);
void res_log_fcritical(const char * fmt, ...);

#ifdef __cplusplus
}
#endif
#endif // RESLOG_H
