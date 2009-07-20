#include <stdbool.h>
#include <util.h>
#include <sched_file.h>
#include <history.h>
#include <sched_kw_wconinje.h>

void shut_injectors(sched_kw_wconinje_type * kw , int restart_nr , void * arg) {
  sched_kw_wconinje_set_surface_flow(kw , "WI_1" , 0);
  sched_kw_wconinje_set_surface_flow(kw , "WI_2" , 0);
}



int main(int argc, char **argv)
{
  if(argc < 2)
  {
    printf("Usage: sched_test.x my_sched_file.SCH\n");
    return 0;
  }
  
  int    num_restart_files;
  int    last_restart_file;

  FILE * stream     = NULL;

  sched_file_type * sched_file = NULL;

  
  // First verify that we can read a SCHEDULE file.
  sched_file = sched_file_alloc(-1);
  printf("-- Loading %s..\n", argv[1]);
  sched_file_parse(sched_file, -1 , argv[1]);
  sched_file_update_blocks(sched_file , 1 , 62 , WCONINJE , shut_injectors , NULL);
  sched_file_fprintf_i(sched_file , 62 , "/tmp/target.SCH");
  
  return 0;
}
