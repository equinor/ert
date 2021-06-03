/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'log.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>

#include "ert/util/build_config.h"

#ifdef HAVE_FSYNC
#include <unistd.h>
#endif

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#include <ert/util/util.hpp>

#include <ert/res_util/log.hpp>

struct log_struct {
  char             * filename;
  FILE             * stream;
  int                fd;
  message_level_type log_level;
  message_level_type log_level_stdout;
  int                msg_count;
  bool               stream_owner;
#ifdef HAVE_PTHREAD
  pthread_mutex_t    mutex;
#endif
};



static void log_delete_empty(const log_type * logh) {
  if (!logh->filename)
    return;

  if (!util_file_exists(logh->filename))
    return;

  if (util_file_size(logh->filename) == 0)
    remove( logh->filename );
}


const char * log_get_filename( const log_type * logh ) {
  return logh->filename;
}

int log_get_msg_count( const log_type * logh) {
  return logh->msg_count;
}

/**
 * If an incoming message is below or equal to the configured log_level, it is included. So a high log_level will
 * include more messages.
 */
void log_set_level( log_type * logh , message_level_type log_level) {
  logh->log_level = log_level;
}



log_type * log_open_stream(FILE * stream, message_level_type log_level) {
  if (!stream)
    return NULL;

  log_type * logh        = (log_type *) util_malloc(sizeof * logh);
  logh->msg_count        = 0;
  logh->log_level        = log_level;
  logh->log_level_stdout = LOG_ERROR;  // non-configurable default
  logh->stream           = stream;
  logh->fd               = fileno( logh->stream );
  logh->filename         = NULL;
  logh->stream_owner     = false;
#ifdef HAVE_PTHREAD
  pthread_mutex_init( &logh->mutex , NULL );
#endif

  return logh;
}


log_type * log_open_file(const char * filename , message_level_type log_level) {
  if (!filename)
    return NULL;

  {
    char * path = util_split_alloc_dirname(filename);
    if (path) {

      if (!util_is_directory(path)) {
        if (!util_mkdir_p(path)) {
          free(path);
          return NULL;
        }
      }

      free(path);
    }
  }


  FILE * stream = fopen( filename, "a+");
  if (!stream)
    return NULL;


  log_type * logh = log_open_stream(stream, log_level);
  if (logh) {
    logh->filename = util_alloc_string_copy(filename);
    logh->stream_owner = true;
  }

  return logh;
}


static bool log_include_message_stdout(const log_type *logh,
                                message_level_type message_level) {
  return message_level >= logh->log_level_stdout;
}

static bool log_include_message(const log_type *logh, message_level_type message_level) {
  return message_level >= logh->log_level;
}

void log_add_message_stream(FILE * stream, bool add_timestamp, message_level_type message_level, const char * message) {
  struct tm time_fields;
  time_t    epoch_time;

  time(&epoch_time);
  util_time_utc(&epoch_time , &time_fields);

  if (message_level >= LOG_CRITICAL)
    fprintf(stream, "CRITICAL: ");
  else if (message_level >= LOG_ERROR)
    fprintf(stream, "ERROR:    ");
  else if (message_level >= LOG_WARNING)
    fprintf(stream, "WARNING:  ");
  else if (message_level >= LOG_INFO)
    fprintf(stream, "INFO:     ");
  else if (message_level >= LOG_DEBUG)
    fprintf(stream, "DEBUG:    ");

  if (add_timestamp) {
    if (message)
      fprintf(stream,"%02d/%02d - %02d:%02d:%02d  %s\n",time_fields.tm_mday, time_fields.tm_mon + 1, time_fields.tm_hour , time_fields.tm_min , time_fields.tm_sec , message);
    else
      fprintf(stream,"%02d/%02d - %02d:%02d:%02d   \n",time_fields.tm_mday, time_fields.tm_mon + 1, time_fields.tm_hour , time_fields.tm_min , time_fields.tm_sec);
  } else
    if (message)
      fprintf(stream, "%s\n", message);


}

void log_add_message(log_type *logh,
                     message_level_type message_level,
                     const char* message) {
  if (log_include_message_stdout(logh, message_level))
    printf("%s\n", message);  // temporary implementation of logging to terminal

  if (!log_include_message(logh, message_level))
    return;

  if (logh->stream == NULL)
    util_abort("%s: logh->stream == NULL - must call log_reset_filename() first \n",__func__);

#ifdef HAVE_PTHREAD
    pthread_mutex_lock( &logh->mutex );
#endif

  log_add_message_stream(logh->stream, true, message_level, message);
  log_sync( logh );
  logh->msg_count++;

#ifdef HAVE_PTHREAD
    pthread_mutex_unlock( &logh->mutex );
#endif
}


void log_sync(log_type * logh) {
#ifdef HAVE_FSYNC
  fsync( logh->fd );
#endif
  fseek( logh->stream , 0 , SEEK_END );
}



void log_close( log_type * logh ) {
  if (logh->stream_owner)
    fclose( logh->stream );  /* This closes BOTH the FILE * stream and the integer file descriptor. */
  log_delete_empty( logh );
  free( logh->filename );
  free( logh );
}
