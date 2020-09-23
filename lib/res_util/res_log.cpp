/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'res_log.c' is part of ERT - Ensemble based Reservoir Tool.

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

/*
 * TODO:
 *
 * Should we have these as macros which can then insert the
 * filename/line-number in the message?
 */

#include <stdarg.h>
#include <string.h>

#include <ert/util/util.hpp>

#include <ert/res_util/res_log.hpp>
#include <ert/res_util/res_util_defaults.hpp>


static FILE * DEFAULT_STREAM = stdout;
static log_type * logh = NULL;               /* Handle to an open log file. */



static void res_log_init_stream(message_level_type log_level,
                                FILE * stream) {
  if (logh)
    log_close(logh);

  logh = log_open_stream(stream, log_level);
  if (!logh)
    fprintf(stderr,"Could not open stderr log stream\n");
}


/**
 * Initializes the log with default filename and default log level.
 */
static void res_log_init_default(){
  res_log_init_stream(DEFAULT_LOG_LEVEL, DEFAULT_STREAM);
}


/**
 * Adding a message with a given message_level.
 *
 * A higher message_level means "more important", as only messages with
 * message_level above the configured log_level will be included.
 */
void res_log_add_message(message_level_type message_level,
                         const char* message) {
  if (!logh)
    res_log_init_default();

  /* We have not managed to open log handle, if the message is critical we write to stderr. */
  if (!logh) {
    if (message_level >= LOG_ERROR )
      log_add_message_stream(stderr, message_level, (message_level_type)false, message);
    return;
  }


  log_add_message(logh, message_level, message);
}


static void res_log_va(message_level_type level, const char * fmt, va_list args) {
  char * message = util_alloc_sprintf_va(fmt, args);
  res_log_add_message(level, message);
  free(message);
}


void res_log_debug(const char* msg) {
  res_log_add_message(LOG_DEBUG, msg);
}

void res_log_info(const char* msg) {
  res_log_add_message(LOG_INFO, msg);
}

void res_log_warning(const char* msg) {
  res_log_add_message(LOG_WARNING, msg);
}

void res_log_error(const char* msg) {
  res_log_add_message(LOG_ERROR, msg);
}

void res_log_critical(const char* msg) {
  res_log_add_message(LOG_CRITICAL, msg);
}


void res_log_fdebug(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  res_log_va(LOG_DEBUG, fmt, ap);
  va_end(ap);
}

void res_log_finfo(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  res_log_va(LOG_INFO, fmt, ap);
  va_end(ap);
}

void res_log_fwarning(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  res_log_va(LOG_WARNING, fmt, ap);
  va_end(ap);
}

void res_log_ferror(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  res_log_va(LOG_ERROR, fmt, ap);
  va_end(ap);
}

void res_log_fcritical(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  res_log_va(LOG_CRITICAL, fmt, ap);
  va_end(ap);
}


/**
 *  The logging uses log_level to determine if an incoming message is to be
 *  included in the log.
 *
 *  A high log_level setting will include more messages.
 *
 *  if log_file_name=NULL the logger will write to DEFAULT_STREAM (i.e. stdout).
 */
bool res_log_init_log(message_level_type log_level,
                      const char * log_file_name,
                      bool verbose) {
  // If a log is already open, close it
  if(logh)
     log_close(logh);

  if(log_file_name)
    logh = log_open_file(log_file_name , log_level );
  else
    logh = log_open_stream( DEFAULT_STREAM, log_level);

  if (!logh) {
    fprintf(stderr,"Failed to open log handle to %s\n", log_file_name);
    return false;
  }


  if (verbose && log_file_name)
    printf("Activity will be logged to %s \n",log_get_filename( logh ));

  log_add_message(logh, LOG_INFO, "ert configuration loaded");
  return true;
}


void res_log_close() {
  if (logh) {
    log_add_message(logh, (message_level_type)false,
                    "Exiting ert application normally - all is fine(?)");
    log_close( logh );
  }
  logh = NULL;
}

const char * res_log_get_filename() {
  if (logh)
    return log_get_filename(logh);
  else
    return NULL;
}

