#include <util.h>
#include <stdlib.h>
#include <sched_file.h>
#include <ecl_util.h>
#include <time.h>
#include <stdio.h>



int main (int argc , char ** argv) {
  time_t start_time;
  if (argc != 3) 
    util_exit("usage: ECLIPSE.DATA SCHEDULE_FILE \n");
  {
    const char * data_file       = argv[1];
    const char * schedule_file   = argv[2];
    sched_file_type * sched_file = sched_file_alloc();
    start_time = ecl_util_get_start_date( data_file );
    sched_file_parse(sched_file , start_time , schedule_file);
    sched_file_summarize(sched_file , stdout);
    sched_file_free(sched_file);
  }
}
