#include <util.h>
#include <sched_file.h>
#include <history.h>

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    printf("Usage: sched_test.x my_sched_file.SCH\n");
    return 0;
  }

  // First verify that we can read a SCHEDULE file.
  int num_restart_files, last_restart_file;
  sched_file_type * sched_file = sched_file_alloc();
  printf("-- Loading %s..\n", argv[1]);
  sched_file_parse(sched_file, -1, argv[1]);



  // Get the nr of restart files and dump history ecl style
  num_restart_files = sched_file_get_nr_restart_files(sched_file);
  last_restart_file = num_restart_files - 1;
  printf("-- Schedule file \"%s\" will create %i restart files.\n", argv[1], num_restart_files);
  printf("-- Writing all steps to \"sched_test_out_01.SCH\"..\n");
  sched_file_fprintf_i(sched_file, last_restart_file, "sched_test_out_01.SCH");



  // Store a binary rep and free the internal rep
  printf("-- Saving binary representation in \"sched_test_stor_01.bin\"...\n");
  FILE * stream = util_fopen("sched_test_stor_01.bin","w");
  sched_file_fwrite(sched_file, stream);
  printf("-- Freeing internal representation..\n");
  sched_file_free(sched_file);
  fclose(stream);



  // Load the binary rep and dump history ecl style to another file
  printf("-- Loading internal representation from \"sched_test_stor_01.bin\"..\n");
  stream = util_fopen("sched_test_stor_01.bin","r");
  sched_file = sched_file_fread_alloc(stream);
  printf("-- Writing all steps to \"sched_test_out_02.SCH\"..\n");
  sched_file_fprintf_i(sched_file, last_restart_file, "sched_test_out_02.SCH");
  fclose(stream);

  // Try to create a history_type object
  printf("-- Creating history object from \"%s\"..\n", argv[1]);
  history_type * history = history_alloc_from_sched_file(sched_file);
  printf("-- Saving binary history object to \"sched_test_stor_02.bin\"\n");
  stream = util_fopen("sched_test_stor_02.bin","w");
  history_fwrite(history, stream);
  printf("-- Freeing internal representation of history..\n");
  history_free(history);
  fclose(stream);

  // Try to load the history_type oject from disk.
  printf("-- Loading history from \"sched_test_stor_02.bin\"..\n");
  stream = util_fopen("sched_test_stor_02.bin", "r");
  history = history_fread_alloc(stream);
  fclose(stream);



  // Clean up
  printf("-- Cleaning up..\n");
  history_free(history);
  sched_file_free(sched_file);



  return 0;
}
