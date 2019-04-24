/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'log_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_LOG_CONFIG_H
#define ERT_LOG_CONFIG_H

#include <ert/config/config_content.hpp>
#include <ert/res_util/log.hpp>

#ifdef __cplusplus
extern "C" {
#endif


#define LOG_CRITICAL_NAME "CRITICAL"
#define LOG_CRITICAL_DEPRECATED_NAME "0"

#define LOG_ERROR_NAME "ERROR"
#define LOG_ERROR_DEPRECATED_NAME "1"

#define LOG_WARNING_NAME "WARNING"
#define LOG_WARNING_DEPRECATED_NAME "2"

#define LOG_INFO_NAME "INFO"
#define LOG_INFO_DEPRECATED_NAME "3"

#define LOG_DEBUG_NAME "DEBUG"
#define LOG_DEBUG_DEPRECATED_NAME "4"


typedef struct log_config_struct log_config_type;

log_config_type * log_config_alloc_load(const char *);
log_config_type * log_config_alloc(const config_content_type *);
log_config_type * log_config_alloc_full(const char * log_file, message_level_type message_level);
void              log_config_free(log_config_type *);

const char *             log_config_get_log_file(const log_config_type *);
const message_level_type log_config_get_log_level(const log_config_type *);

message_level_type log_config_level_parser(const char * level);

#ifdef __cplusplus
}
#endif
#endif
