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
#include <ecl_diag.h>
#include <history.h>


int main(int argc, char ** argv) {
  int files;
  const char *path    = "/d/proj/bg/frs/EnKF_3Dsynthetic/FourWellsEnKF/Refcase";
  const char *base    = "ECLIPSE";
  char ** file_list   = ecl_util_alloc_scandir_filelist(path , base , ecl_summary_file , false , &files);
  char  * header_file = ecl_util_alloc_exfilename(path , base , ecl_summary_header_file , false , -1);
  
  {
    history_type * hist;
    ecl_sum_type * ecl_sum = ecl_sum_fread_alloc_interactive(true);
    hist = history_alloc_from_summary(ecl_sum , false);
    history_summarize(hist);
    history_free(hist);
    ecl_sum_free(ecl_sum);
  }

  util_free_string_list(file_list , files);
  free(header_file);
}
