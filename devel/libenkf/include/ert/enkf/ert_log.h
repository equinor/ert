#ifndef ERTLOG_H
#define ERTLOG_H

#include <stdio.h>
#include <stdbool.h>
#include <ert/util/log.h>
#include <ert/util/util.h>

void ert_log_init_log(int log_level,const char * log_file_name,const char * user_log_file_name,bool verbose);
void ert_log_add_fmt_message(int message_level , FILE * dup_stream , const char * fmt , ...);
void ert_log_add_message(int message_level , FILE * dup_stream , char* message, bool free_message);
void ert_log_close();
bool ert_log_is_open();
int ert_log_get_log_level();
char * ert_log_get_filename();
#endif // ERTLOG_H
