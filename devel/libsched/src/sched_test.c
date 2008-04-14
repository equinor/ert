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
  int files;
  const char *base_dir     = "tmpdir_";
  const char *base     	   = "ECLIPSE";
  const char *history_file = "/d/proj/bg/frs/EnKF_3Dsynthetic/FourWellsEnKF/Observations/History";
  /*
    char ** file_list    = ecl_util_alloc_scandir_filelist(path , base , ecl_summary_file , false , &files);
    char  * header_file  = ecl_util_alloc_exfilename(path , base , ecl_summary_header_file , false , -1);
  */
  
  
  {
    history_type * hist;
    {
      FILE * stream = util_fopen(history_file , "r");
      hist = history_fread_alloc(stream);
      fclose(stream);
      history_summarize(hist , stdout);
    }
    history_ens_diag_ens_interactive(base_dir , "ECLIPSE" , false , false , true , hist);
  }

  /*
    util_free_string_list(file_list , files);
    free(header_file);
  */
}
				  
