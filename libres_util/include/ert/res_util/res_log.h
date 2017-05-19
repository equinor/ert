/*
   Copyright (C) 2017  Statoil ASA, Norway.
    
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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>

#include <ert/util/log.h>

void res_log_init_log(int log_level,const char * log_file_name, bool verbose);
void res_log_init_log_default_log_level(const char * log_file_name, bool verbose);
void res_log_init_log_default( bool verbose);
void res_log_add_fmt_message(int message_level , FILE * dup_stream , const char * fmt , ...);
void res_log_add_message(int message_level , FILE * dup_stream , char* message, bool free_message);
void res_log_add_message_py(int message_level, char* message);
void res_log_close();
bool res_log_is_open();
int res_log_get_log_level();
const char * res_log_get_filename();
log_type * res_log_get_logh();
void res_log_open_empty();

#ifdef __cplusplus
}
#endif
#endif // RESLOG_H
