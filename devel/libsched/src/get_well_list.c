#include <stdbool.h>
#include <util.h>
#include <sched_file.h>
#include <history.h>

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    printf("Usage: get_well_list.x my_sched_file.SCH\n");
    return 0;
  }
  
  int     num_wells;
  char ** well_list;

  sched_file_type * sched_file = NULL;
  history_type    * history    = NULL;


  sched_file = sched_file_alloc(-1);
  printf("-- Loading %s..\n", argv[1]);
  sched_file_parse(sched_file, -1 , argv[1]);


  printf("-- Creating history object from \"%s\"..\n", argv[1]);
  history = history_alloc_from_sched_file(sched_file);

  well_list = history_alloc_well_list(history, &num_wells);

  for(int well_nr = 0; well_nr < num_wells; well_nr++)
    printf("%s\n", well_list[well_nr]);

  printf("-- Cleaning up..\n");
  history_free(history);
  sched_file_free(sched_file);
  util_free_stringlist(well_list, num_wells);

  return 0;
}
