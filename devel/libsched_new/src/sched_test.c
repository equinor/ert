#include <sched_file.h>
#include <util.h>

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    printf("Usage: sched_test.x my_sched_file.SCH\n");
    return 0;
  }

  // First verify that we can read a SCHEDULE file.
  int num_report_steps;
  sched_file_type * sched_file = sched_file_alloc();
  printf("-- Loading %s..\n", argv[1]);
  sched_file_parse(sched_file, -1, argv[1]);

  // Get the nr of report steps and dump history ecl style
  num_report_steps = sched_file_get_nr_report_steps(sched_file);
  printf("-- Schedule file \"%s\" had %i report steps.\n", argv[1], num_report_steps);
  printf("-- Writing all report steps to \"sched_test_out_01.SCH\"..\n");
  sched_file_fprintf(sched_file, num_report_steps, "sched_test_out_01.SCH");

  // Store a binary rep and free the internal rep
  printf("-- Saving binary representation in \"sched_test_stor.bin\"...\n");
  FILE * stream = util_fopen("sched_test_stor.bin","w");
  sched_file_fwrite(sched_file, stream);
  printf("-- Freeing internal representation..\n");
  sched_file_free(sched_file);
  fclose(stream);


  // Load the binary rep and dump history ecl style to another file
  printf("-- Loading internal representation from \"sched_test_stor.bin\"..\n");
  stream = util_fopen("sched_test_stor.bin","r");
  sched_file = sched_file_fread_alloc(stream);
  printf("-- Writing all report steps to \"sched_test_out_02.SCH\"..\n");
  sched_file_fprintf(sched_file, num_report_steps, "sched_test_out_02.SCH");
  printf("-- Cleaning up..\n");
  sched_file_free(sched_file);
  fclose(stream);

  return 0;
}
