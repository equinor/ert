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
 * - Should we duplicate messages to stderr on errors? It should probably not be the callers job to decide as it is now.
 * -- Today this can be done with res_log_add_message or res_log_add_fmt_message. Erik wants to change the first
 * to no longer accept a secondary stream, so then res_log_add_fmt_message is the only option.
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
 * The logging uses log_level to determine if an incoming message is to be included in the log.
 * A high log_level setting will include more messages.
 *
 * if log_file_name=NULL then DEFAULT_LOG_FILE is used
 */
void res_log_init_log( message_level_type log_level , const char * log_file_name, bool verbose){
  // If a log is already open, close it
  if(logh)
     log_close(logh);

  if(log_file_name==NULL){
    log_file_name=DEFAULT_LOG_FILE;
  }
  logh = log_open(log_file_name , log_level );

  if (verbose)
    printf("Activity will be logged to ..............: %s \n",log_get_filename( logh ));
  log_add_message(logh , LOG_INFO , NULL , "ert configuration loaded" , false);
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

/**
 * Simplified method for adding a message to the log. Does not duplicate the message, and does not
 * attempt to free the message.
 * @param message_level Importance of the message.
 * @param message The message, will not be free'd
 */
void res_log_add_message_str(message_level_type message_level, const char* message){
  if(logh==NULL)
    res_log_init_log_default(true);
  log_add_message_str(logh, message_level, message);
}

void res_log_add_message_py(int message_level, char* message){
    res_log_add_message(message_level, NULL, message, false);
}

/**
 * Adding a message with a given message_level.
 *
 * A low message_level means "more important", as only messages with
 * message_level below the configured log_level will be included.
 */
void res_log_add_message(message_level_type message_level , FILE * dup_stream , char* message, bool free_message) {
   if(logh==NULL)
     res_log_init_log_default(true);
   log_add_message(logh, message_level, dup_stream, message, free_message);
}

/**
 * Adding a message with a given message_level.
 *
 * Expects fmt to be a string with potential string-formatting, and "..." is the
 * required arguments to the string-formatting.
 */
void res_log_add_fmt_message(message_level_type message_level , FILE * dup_stream , const char * fmt , ...) {
  if(logh==NULL)
    res_log_init_log_default(true);

  if (log_include_message(logh,message_level)) {
      char * message;
      va_list ap;
      va_start(ap , fmt);
      message = util_alloc_sprintf_va( fmt , ap );
      log_add_message( logh , message_level , dup_stream , message , true);
      va_end(ap);
    }
}


void res_log_close(){
    if (log_is_open( logh ))
      log_add_message( logh , false , NULL , "Exiting ert application normally - all is fine(?)" , false);
    log_close( logh );
    logh = NULL;
}

bool res_log_is_open(){
    if(logh==NULL)
        return false;
    return log_is_open(logh);
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

log_type * res_log_get_logh() {
  if(logh==NULL)
    res_log_init_log_default(true);
  return logh;
}

void res_log_open_empty() {
  if(logh)
    log_close(logh);
  logh = log_open(NULL, DEFAULT_LOG_LEVEL);
}
