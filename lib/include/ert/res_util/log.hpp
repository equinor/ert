/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'log.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_LOG_H
#define ERT_LOG_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

//Same as pythons default log levels, but with different numeric values.
typedef enum {
  // A serious error, indicating that the program itself may be unable to
  // continue running.
  LOG_CRITICAL = 50,

  // Due to a more serious problem, the software has not been able to perform
  // some function.
  LOG_ERROR    = 40,

  // An indication that something unexpected happened, or indicative of some
  // problem in the near future (e.g. "disk space low"). The software is still
  // working as expected.
  LOG_WARNING  = 30,

  // Confirmation that things are working as expected.
  LOG_INFO     = 20,

  // Detailed information, typically of interest only when diagnosing problems.
  LOG_DEBUG    = 10
} message_level_type;


typedef struct log_struct log_type;
  log_type   * log_open_file(const char *filename, message_level_type log_level);
  log_type   * log_open_stream(FILE * stream, message_level_type log_level);

  void         log_add_message_stream(FILE * stream, bool add_timestamp, message_level_type message_level, const char * message);
  void         log_add_message(log_type *logh, message_level_type message_level,  const char* message);
  void         log_set_level( log_type * logh , message_level_type new_level);
  void         log_close( log_type * logh );
  void         log_sync(log_type * logh);
  const char * log_get_filename( const log_type * logh );
  void         log_set_level( log_type * logh , message_level_type log_level);
  int          log_get_msg_count(const log_type * logh);
  message_level_type log_get_level( const log_type * logh);
  message_level_type log_get_level( const log_type * logh);

#ifdef __cplusplus
}
#endif

#endif
