/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'log_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/res_util/res_util_defaults.hpp>
#include <ert/util/util.h>

#include <ert/enkf/log_config.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/enkf_defaults.hpp>

struct log_config_struct {

  char * log_file;
  message_level_type message_level;

};


static void log_config_init(log_config_type *, const config_content_type *);


static log_config_type * log_config_alloc_default() {
  log_config_type * log_config = (log_config_type *)util_malloc(sizeof * log_config);

  log_config->log_file = util_alloc_string_copy(DEFAULT_LOG_FILE);
  log_config->message_level = DEFAULT_LOG_LEVEL;

  return log_config;
}


log_config_type * log_config_alloc_load(const char * config_file) {
  config_parser_type * config_parser = config_alloc();
  config_content_type * config_content = NULL;
  if(config_file)
    config_content = model_config_alloc_content(config_file, config_parser);

  log_config_type * log_config = log_config_alloc(config_content);

  config_content_free(config_content);
  config_free(config_parser);

  return log_config;
}


log_config_type * log_config_alloc(const config_content_type * config_content) {
  log_config_type * log_config = log_config_alloc_default();

  if(config_content)
    log_config_init(log_config, config_content);

  return log_config;
}


const char * log_config_get_log_file(const log_config_type * log_config) {
  return log_config->log_file;
}


const message_level_type log_config_get_log_level(
        const log_config_type * log_config
        ) {

  return log_config->message_level;
}


static void log_config_set_log_file(
        log_config_type * log_config,
        const char * log_file
        ) {

  free(log_config->log_file);
  log_config->log_file = util_alloc_string_copy(log_file);
}


/**
 * This method parses the 'LOG_LEVEL_KEY' value according to the following
 * rules:
 *
 * - If it is an integer 0 <= i <= 4 then it issues an deprecation warning and
 *   uses the approximately correct enum of message_level_type.
 *
 * - If it is one of the strings CRITICAL, ERROR, WARNING, INFO, DEBUG it
 *   returns the corresponding enum of message_level_type
 *
 * - Else it returns the DEFAULT_LOG_LEVEL
 */
message_level_type log_config_level_parser(const char * level) {

  typedef struct {
    const char * log_keyword; // The keyword written in the config file
    const char * log_old_numeric_str; // The *old* integer value
    const message_level_type log_enum; // The enum for the new log-level
  } log_tupple;

  const int nr_of_log_levels = 5;
  log_tupple log_levels[] = {
  //           New name           Deprecated name            Corresponding enum
          { LOG_CRITICAL_NAME, LOG_CRITICAL_DEPRECATED_NAME, LOG_CRITICAL },
          { LOG_ERROR_NAME,    LOG_ERROR_DEPRECATED_NAME,    LOG_ERROR    },
          { LOG_WARNING_NAME,  LOG_WARNING_DEPRECATED_NAME,  LOG_WARNING  },
          { LOG_INFO_NAME,     LOG_INFO_DEPRECATED_NAME,     LOG_INFO     },
          { LOG_DEBUG_NAME,    LOG_DEBUG_DEPRECATED_NAME,    LOG_DEBUG    }
          };


  for (int i = 0; i < nr_of_log_levels; i++) {
    log_tupple curr_log_level = log_levels[i];

    // We found a new proper name
    if (strcmp(level, curr_log_level.log_keyword)==0)
      return curr_log_level.log_enum;

    // We found an old integer level
    else if (strcmp(level, curr_log_level.log_old_numeric_str)==0) {
      fprintf(stderr,
              "** Deprecation warning: Use of %s %s is deprecated, use %s %s instead\n",
              LOG_LEVEL_KEY, curr_log_level.log_old_numeric_str,
              LOG_LEVEL_KEY, curr_log_level.log_keyword
              );

      return curr_log_level.log_enum;
    }
  }

  fprintf(stderr, "** The log_level: %s is not valid, using default log level\n", level);
  return DEFAULT_LOG_LEVEL;
}


static void log_config_init(
        log_config_type * log_config,
        const config_content_type * content
        ) {

  if (config_content_has_item(content, LOG_FILE_KEY)) {
    const char * log_file = config_content_get_value_as_abspath(content, LOG_FILE_KEY);
    log_config_set_log_file(log_config, log_file);
  }

  // If no log file, make default log file relative to config file
  else if(!util_is_abs_path(log_config->log_file)) {
    char * working_dir;
    util_alloc_file_components(
            config_content_get_config_file(content, true),
            &working_dir, NULL, NULL);

    char * abs_default_log_file = util_alloc_filename(working_dir, log_config->log_file, NULL);

    log_config_set_log_file(log_config, abs_default_log_file);

    free(working_dir);
    free(abs_default_log_file);
  }

  if(config_content_has_item(content, LOG_LEVEL_KEY)) {
    const char * log_level_str = config_content_get_value(content, LOG_LEVEL_KEY);
    log_config->message_level = log_config_level_parser(log_level_str);
  }
}


void log_config_free(log_config_type * log_config) {
  if(!log_config)
    return;

  free(log_config->log_file);
  free(log_config);
}
