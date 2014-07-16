#include <ert/enkf/ertlog.h>
#include <ert/util/util.h>

log_type             * logh;               /* Handle to an open log file. */


void ertlog_init_log( enkf_main_type * enkf_main , const config_type * config ) {
  logh = log_open( NULL , DEFAULT_LOG_LEVEL );

  if (config_item_set( config , LOG_LEVEL_KEY))
    log_set_level(logh, config_get_value_as_int(config , LOG_LEVEL_KEY));

  if (config_item_set( config , LOG_FILE_KEY))
    log_reopen( logh , config_get_value(config , LOG_FILE_KEY));
  else {
    char * log_file = util_alloc_filename(NULL , enkf_main->user_config_file , DEFAULT_LOG_FILE);
    log_reopen( logh , log_file );
    free( log_file );
  }

  if (enkf_main->verbose)
    printf("Activity will be logged to ..............: %s \n",log_get_filename( logh ));
  log_add_message(logh , 1 , NULL , "ert configuration loaded" , false);
}


void ertlog_add_fmt_message(int message_level , FILE * dup_stream , const char * fmt , ...) {
    if (log_include_message(logh,message_level)) {
      char * message;
      va_list ap;
      va_start(ap , fmt);
      message = util_alloc_sprintf_va( fmt , ap );
      log_add_message( logh , message_level , dup_stream , message , true);
      va_end(ap);
    }
}
