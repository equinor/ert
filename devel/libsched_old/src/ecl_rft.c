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
#include <ecl_rft_node.h>
#include <sched_file.h>
#include <sched_util.h>


int main(int argc, char ** argv) {
  const bool order_tvd_p = false;
  char * config_file = argv[1];
  if (!util_file_exists(config_file)) {
    fprintf(stderr,"Could not open config-file:%s - aborting \n",config_file);
    abort();
  }
  {
    FILE * stream = fopen(config_file , "r");
    char * obs_path;
    sched_file_type * sched_file;
    ecl_rft_vector_type * rft_vector;
    bool at_eof = false;
    
    {
      char * eclipse_rft_file = util_fscanf_alloc_token(stream);  util_forward_line(stream , &at_eof);
      char * sched_file_name  = util_fscanf_alloc_token(stream);  util_forward_line(stream , &at_eof);
      sched_file              = sched_file_alloc( 0 );
      sched_file_parse(sched_file , sched_file_name);
      rft_vector 	      = ecl_rft_vector_alloc(eclipse_rft_file , true);
      printf("rft_file.......: %s \n",eclipse_rft_file);
      printf("Schedule file..: %s \n",sched_file_name);
      free(eclipse_rft_file);
      free(sched_file_name);
    }
    
    obs_path = util_fscanf_alloc_token(stream); util_forward_line(stream , &at_eof);
    printf("obs_path.......: %s \n\n",obs_path);
    {
      char * obs_file = malloc(strlen(obs_path) + 10); 
      while (1) {
	char * well                         = util_fscanf_alloc_token(stream);
	if (well != NULL) {
	  const ecl_rft_node_type * rft_node  = ecl_rft_vector_get_node(rft_vector , well);
	  char * tvd_file                     = util_fscanf_alloc_token(stream);
	  int    report_step                  = sched_file_time_t_to_report_step(sched_file , ecl_rft_node_get_recording_time(rft_node));
	  
	  util_forward_line(stream , &at_eof);
	  sprintf(obs_file , "%s/%04d/RFT" , obs_path , report_step);
	  printf("Blocking well: %-8s %-54s ->  %s\n",well,tvd_file,obs_file);
	  /*
         ecl_rft_vector_fprintf_rft_obs(rft_vector , 10.0 , well , order_tvd_p , tvd_file , obs_file , 1.0);
      */
	  free(well);
	  free(tvd_file);
	  ecl_rft_node_export_DEPTH(rft_node , "/d/proj/bg/enkf/EnKF_OS2/Static/RFT");
	} else break;
      }
      free(obs_file);
    }
    free(obs_path);
    ecl_rft_vector_free(rft_vector);
    sched_file_free(sched_file);
    fclose(stream);
  }
  return 0;
}
