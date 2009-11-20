#include <stdlib.h>
#include <stdio.h>
#include <rms_file.h>
#include <rms_tagkey.h>
#include <rms_stats.h>
#include <list.h>


int main (int argc , char **argv) {
  int i;

  argc--;
  argv++;

  for (i = 0; i < argc; i++) {
    rms_file_type *file = rms_file_alloc(argv[i] , false);
    rms_file_fread(file);
    rms_file_printf(file , stdout);
    rms_file_free(file);
  }

  return 0;
}

