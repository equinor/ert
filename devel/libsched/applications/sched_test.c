#include <stdbool.h>
#include <util.h>
#include <sched_file.h>
#include <history.h>
#include <sched_kw_wconinje.h>
#include <sched_history.h>
#include <well_history.h>


int main(int argc, char **argv)
{
  time_t start_time;
  util_make_date(1,1,2000);
  sched_file_type * sched_file = sched_file_parse_alloc( argv[1] , start_time);
  sched_history_type * sched_history = sched_history_alloc();
  sched_history_update( sched_history , sched_file );
  sched_history_install_index( sched_history );

  printf("%g \n",sched_history_iget( sched_history , "WOPRH:OP_1" , 30));

  sched_history_free( sched_history );
  sched_file_free( sched_file );
}
