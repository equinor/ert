#include <stdlib.h>
#include <ecl_kw.h>
#include <fortio.h>
#include <util.h>
#include <string.h>
#include <ecl_util.h>
#include <ecl_sum.h>
#include <hash.h>
#include <stdbool.h>
#include <ecl_rft_vector.h>
#include <ecl_grid.h>
#include <history.h>
#include <history_ens_diag.h>



int main(int argc, char ** argv) {
  history_type * hist;

  if(argc < 2)
  {
    printf("usage: sched.x schedule_file.SCH\n");
    return 0;
  }

  sched_file_type * sched_file = sched_file_alloc( util_make_time1(1 , 1 , 2000) );
  sched_file_parse(sched_file ,  argv[1]);
  
  printf("nr report steps: %i\n", sched_file_count_report_steps(sched_file));

  
  hist = history_alloc_from_schedule( sched_file );
  sched_file_fprintf(sched_file , 25 , -1 , -1 , "/tmp/target.SCH");
  
  history_free( hist );
  sched_file_free(sched_file);
}
				  
