#include <stdlib.h>
#include <stdio.h>
#include <sched_file.h>
#include <stdio.h>
#include <util.h>
#include <history.h>
#include <ecl_sum.h>
#include <ecl_util.h>

int main(int argc , char **argv) {
  FILE * stream;
  sched_file_type *s;
  history_type *h;
  
  s = sched_file_alloc((const int [3]) {1 , 1 , 1999});
  sched_file_parse(s , "SCHEDULE_orig.INC");
  
  
  h = history_alloc_from_schedule(s);
  stream = util_fopen("/h/a152128/EnKF_OE/Run3/Observations/History" , "w");
  history_fwrite(h , stream);
  fclose(stream);
  history_free(h);

  stream = util_fopen("/h/a152128/EnKF_OE/Run3/Observations/History" , "r");
  h = history_fread_alloc(stream);
  history_free(h);

  return 0;
}
				  
