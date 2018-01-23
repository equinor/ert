/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
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

/*TODO:
 * For libres:
 * - Go over the current log-calls and make sure they use reasonable log-levels
 * For libecl:
 * - Change log.h to use the normal semantics of log-levels, where higher means more important message.
 * - Change log.h to use the same numeric values for log-levels as Python
 * - Should we insert ERROR/WARNING etc in the front of the messages?
 * - Should we actually have these as macros which can then insert the filename/line-number in the message?
 * For both:
 * - Change all occurrences of ints to represent log-level to actually use the message_level_type?
 */

#include <stdarg.h>
#include <string.h>

#include <ert/util/util.h>

#include <ert/res_util/res_log.h>
#include <ert/res_util/res_util_defaults.h>

static log_type * logh = NULL;               /* Handle to an open log file. */



/**
 * Adding a message with a given message_level.
 *
 * A higher message_level means "more important", as only messages with
 * message_level above the configured log_level will be included.
 */
void res_log_add_message(message_level_type message_level,
                         const char* message) {
  if (!logh)
    res_log_init_log_default(true);
  log_add_message(logh, message_level, NULL, message);
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
 * The logging uses log_level to determine if an incoming message is to be included in the log.
 * A high log_level setting will include more messages.
 *
 * if log_file_name=NULL then DEFAULT_LOG_FILE is used
 */
void res_log_init_log(message_level_type log_level,
                      const char * log_file_name,
                      bool verbose) {
  // If a log is already open, close it
  if(logh)
     log_close(logh);

  if(log_file_name==NULL){
    log_file_name=DEFAULT_LOG_FILE;
  }
  logh = log_open(log_file_name , log_level );

  if (verbose)
    printf("Activity will be logged to ..............: %s \n",log_get_filename( logh ));
  log_add_message(logh, LOG_INFO, NULL, "ert configuration loaded");
}

/**
 * Initializes the log with log level DEFAULT_LOG_LEVEL.
 * If log_file_name=NULL then DEFAULT_LOG_FILE is used
 */
void res_log_init_log_default_log_level(const char * log_file_name, bool verbose){
  res_log_init_log(DEFAULT_LOG_LEVEL,log_file_name,verbose);
}


/**
 * Initializes the log with default filename and default log level.
 */
void res_log_init_log_default( bool verbose){
  res_log_init_log(DEFAULT_LOG_LEVEL, DEFAULT_LOG_FILE,verbose);
}


void res_log_close() {
    if (log_is_open(logh))
      log_add_message(logh, false, NULL, "Exiting ert application normally - "
                                          "all is fine(?)");
    log_close( logh );
    logh = NULL;
}

void res_log_set_log_level(message_level_type log_level){
    if(logh==NULL)
      res_log_init_log_default(true);
    log_set_level(logh, log_level);
}

int res_log_get_log_level(){
  if(logh==NULL)
    res_log_init_log_default(true);
  return log_get_level(logh);
}

const char * res_log_get_filename() {
  if(logh==NULL)
    res_log_init_log_default(true);
  return log_get_filename(logh);
}

void res_log_open_empty() {
  if(logh)
    log_close(logh);
  logh = log_open(NULL, DEFAULT_LOG_LEVEL);
}
