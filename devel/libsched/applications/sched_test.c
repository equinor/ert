#include <stdbool.h>
#include <util.h>
#include <sched_file.h>
#include <history.h>
#include <sched_kw_wconinje.h>




int main(int argc, char **argv)
{
  time_t start_time;
  util_make_date(1,1,2000);
  sched_file_type * sched_file = sched_file_parse_alloc( argv[1] , start_time);
  printf("File: %s contains:%d report steps \n",argv[1] , sched_file_step_count( argv[1] ));
  sched_file_free( sched_file );
}
