#include <stdlib.h>
#include <util.h>
#include <time.h>
#include <string.h>
#include <sched_file.h>
#include <ecl_fstate.h>
#include <ecl_sum.h>
#include <history.h>


static const bool    ENDIAN_CONVERT  = true;
static history_type *GLOBAL_HISTORY  = NULL;

/*****************************************************************/


void sched_parse_wconhist__(const char * _schedule_dump_file , const int * schedule_dump_len , 
			    const char * _obs_path           , const int * obs_path_len      ,
			    const char * _obs_file           , const int * obs_file_len) {
  sched_file_type *s;
  char * schedule_file = util_alloc_cstring(_schedule_dump_file , schedule_dump_len);
  char * obs_path      = util_alloc_cstring(_obs_path , obs_path_len);
  char * obs_file      = util_alloc_cstring(_obs_file , obs_file_len);
  
  {
    FILE * stream = util_fopen(schedule_file , "r");
    s = sched_file_fread_alloc( stream , -1  , -1 , -1.0);
    fclose(stream);
  }
  sched_file_fprintf_rates(s , obs_path , obs_file);  
  sched_file_free(s);
  
  free(schedule_file);
  free(obs_path);
  free(obs_file);
}



/* 
   Fortran input is typically double - but internally this is done with float,
   since that is how perm is stored in eclipse. Consistent??
*/

static void sched_update_compdat_fprintf_static(const char  * _schedule_dump_file , const int * schedule_dump_len , 
						const char  * _schedule_file      , const int * schedule_file_len,
						const char  * _last_date_str      , const int * last_date_len , 
						const double * permx , const double * permz, const int * index_map , const int * nactive) {
  
  sched_file_type *s;
  char * schedule_file      = util_alloc_cstring(_schedule_file      , schedule_file_len);
  char * schedule_dump_file = util_alloc_cstring(_schedule_dump_file , schedule_dump_len);
  char * last_date_str      = util_alloc_cstring(_last_date_str      , last_date_len);

  {
    FILE * stream = util_fopen(schedule_dump_file , "r");
    s = sched_file_fread_alloc( stream , atoi(last_date_str)  , -1 , -1.0);
    fclose(stream);
  }

  if (permx != NULL) {
    int size        = *nactive;
    float * permx_4 = malloc(size * sizeof * permx_4);
    float * permz_4 = malloc(size * sizeof * permz_4);
    if (permz_4 == NULL) {
      fprintf(stderr,"%s: failed to malloc - aborting \n",__func__);
      abort();
    }
    util_double_to_float(permx_4 , permx   , size);
    util_double_to_float(permz_4 , permz   , size);
    sched_file_set_conn_factor(s , permx_4 , permz_4 , index_map );
    free(permx_4);
    free(permz_4);
  }
  
  sched_file_fprintf(s , atoi(last_date_str) , -1 , -1.0 , schedule_file);
  sched_file_free(s);

  free(last_date_str);
  free(schedule_dump_file);
  free(schedule_file);
}

     




void sched_fprintf__(const char * _schedule_dump_file , const int * schedule_dump_len , 
		     const char * _schedule_file      , const int * schedule_file_len,
		     const char * _last_date_str      , const int * last_date_len) {

  sched_update_compdat_fprintf_static(   _schedule_dump_file ,   schedule_dump_len , 
					 _schedule_file      ,   schedule_file_len,
					 _last_date_str      ,   last_date_len , 
					 NULL , NULL , NULL , NULL);
  

}


void sched_update_compdat_fprintf__(const char * _schedule_dump_file , const int * schedule_dump_len , 
				    const char * _schedule_file      , const int * schedule_file_len,
				    const char * _last_date_str      , const int * last_date_len , 
				    const double * permx , const double * permy, const int * index_map, const int * nactive) {

  sched_update_compdat_fprintf_static(   _schedule_dump_file ,   schedule_dump_len , 
					 _schedule_file      ,   schedule_file_len,
					 _last_date_str      ,   last_date_len , 
					 permx ,    permy,   index_map , nactive);
  
}



void sched_load_hist_from_schedule__(const char * _schedule_dump_file , const int * schedule_dump_len , 
                                     const char * _hist_file , const int * hist_file_len) {

  FILE * stream;
  char * schedule_dump_file = util_alloc_cstring(_schedule_dump_file , schedule_dump_len);
  char * hist_file          = util_alloc_cstring(_hist_file , hist_file_len);
  sched_file_type * s;
  stream = util_fopen(schedule_dump_file , "r");
  s    = sched_file_fread_alloc(stream , -1 , -1 , -1.0);
  fclose(stream);

  if (GLOBAL_HISTORY != NULL) 
    history_free(GLOBAL_HISTORY);

  GLOBAL_HISTORY = history_alloc_from_schedule(s);
  stream = util_fopen(hist_file , "w");
  history_fwrite(GLOBAL_HISTORY  , stream);
  fclose(stream);

  sched_file_free(s);
  free(hist_file);
  free(schedule_dump_file);
}



 

void sched_init__(const char * _schedule_file  	   , const int * schedule_file_len,
		  const char * _schedule_dump_file , const int * schedule_dump_len,
		  const char * _init_file          , const int * init_file_len,
		  const char * _hist_file          , const int * hist_file_len,
		  const int  * start_date , 
		  const int  * endian_flip_int , 
		  const int  * index_map) {

  char * schedule_file      = util_alloc_cstring(_schedule_file , schedule_file_len);
  char * schedule_dump_file = util_alloc_cstring(_schedule_dump_file , schedule_dump_len);
  char * hist_file          = util_alloc_cstring(_hist_file , hist_file_len);
  bool   update             = true;

  if (util_file_exists(schedule_dump_file)) 
    update = util_file_update_required(schedule_file , schedule_dump_file);
  
  update = true;
  if (update) {
    char * init_file   = util_alloc_cstring(_init_file , init_file_len);
    bool   endian_flip = util_intptr_2bool(endian_flip_int);
    sched_file_type *s = sched_file_alloc(start_date);
    sched_file_parse(s  , schedule_file);
    sched_file_init_conn_factor(s , init_file , endian_flip , index_map);
    sched_file_fprintf_days_dat(s , "days.dat");

    {
      FILE *stream = util_fopen(schedule_dump_file , "w");
      sched_file_fwrite(s , stream);
      fclose(stream);
    }

    if (util_file_exists(hist_file)) {
      FILE * stream = util_fopen(hist_file , "r");
      GLOBAL_HISTORY = history_fread_alloc(stream);
      fclose(stream);
    } else 
      sched_load_hist_from_schedule__(_schedule_dump_file , schedule_dump_len , _hist_file , hist_file_len);
    
    sched_file_free(s);
    free(init_file);
  }
  
  free(schedule_dump_file);
  free(schedule_file);
  free(hist_file);
}
  

void sched_load_hist_from_summary__(const char * _hist_file , const int * hist_file_len) {
  const bool history_mode = false;
  char * hist_file        = util_alloc_cstring(_hist_file , hist_file_len);
  ecl_sum_type * ecl_sum;
  ecl_sum           = ecl_sum_fread_alloc_interactive(ENDIAN_CONVERT);

  if (ecl_sum == NULL) {
    fprintf(stderr,"%s: sorry returned with invalid ecl_sum object - aborting \n",__func__);
    abort();
  }

  if (GLOBAL_HISTORY != NULL) 
    history_free(GLOBAL_HISTORY);
  GLOBAL_HISTORY = history_alloc_from_summary(ecl_sum , history_mode);
  ecl_sum_free(ecl_sum);

  {
    FILE * stream = util_fopen(hist_file , "w");
    history_fwrite(GLOBAL_HISTORY  , stream);
    fclose(stream);
  }
  
  free(hist_file);
}
 
 
 


/*****************************************************************/


static ecl_sum_type * ecl_diag_avg_load(const char * eclbase_dir , const char * eclbase_name , bool fmt_file , bool unified , bool endian_convert) {
  char *spec_file;
  ecl_sum_type *sum;
  int fmt_mode;
  bool report_mode = false;

  if (fmt_file) 
    fmt_mode = ECL_FORMATTED;
  else 
    fmt_mode = ECL_BINARY;

  spec_file = ecl_util_alloc_exfilename(eclbase_dir , eclbase_name , fmt_file , -1 , ecl_summary_header_file);
  if (unified) {
    char * unif_file = ecl_util_alloc_filename( eclbase_dir , eclbase_name , fmt_file , -1 , ecl_unified_summary_file);
    sum = ecl_sum_fread_alloc(spec_file , 1 , (const char **) &unif_file , report_mode , endian_convert);
    free(unif_file);
  } else {
    int files;
    char **fileList;
    fileList  = ecl_util_alloc_scandir_filelist(eclbase_dir , eclbase_name , ecl_summary_file , fmt_file , &files);
    sum       = ecl_sum_fread_alloc(spec_file , files , (const char **) fileList , report_mode , endian_convert);
    util_free_string_list(fileList , files);
  }
  free(spec_file);
  return sum;
}


static void ecl_diag_avg_production(const char *out_path , const history_type * hist , const char * eclbase_dir , const char * eclbase_avg , const char * eclbase_std , int nwell , const char **well_list , int nvar , const char **var_list , bool fmt_file , bool unified , bool endian_convert) {
  ecl_sum_type *avg = ecl_diag_avg_load(eclbase_dir , eclbase_avg , fmt_file , unified, endian_convert);
  ecl_sum_type *std = ecl_diag_avg_load(eclbase_dir , eclbase_std , fmt_file , unified, endian_convert);
  int iwell, ivar , size;

  size = ecl_sum_get_size(avg);
  printf("size: %d \n",size);
  for (iwell = 0; iwell < nwell; iwell++) {
    if (history_has_well(hist , well_list[iwell])) {
      FILE *stream;
      char * well_file = malloc(strlen(out_path) + 1 + strlen(well_list[iwell]) + 1);
      const char *well = well_list[iwell];
      int istep;
    
      sprintf(well_file , "%s/%s" , out_path , well_list[iwell]);
      stream = util_fopen(well_file , "w");
      for (istep = 0; istep < size; istep++) {
	double time_value;
	time_value = ecl_sum_iget2(avg , istep , 0 );
	fprintf(stream , "%04d  %8.2f" , istep , time_value);
	for (ivar = 0; ivar < nvar; ivar++) {
	  double history_value , avg_value , std_value;
	  int index;
	  const char *var = var_list[ivar];
	  
	  history_value = history_get(hist , istep + 1 , well , var);
	  avg_value = ecl_sum_iget1(avg , istep , well ,  var , &index);
	  std_value = ecl_sum_iget1(std , istep , well ,  var , &index);
	  
	  fprintf(stream , "%16.7f   %16.7f  %16.7f " , history_value , avg_value , std_value);
	}
	fprintf(stream , "\n");
      }
      fclose(stream);
      free(well_file);
    }
  }

  ecl_sum_free(avg);
  ecl_sum_free(std);
}  




static char ** fread_alloc_wells(const char *well_file , int *_nwell) {
  FILE *fileH;
  char **well_list = NULL;
  char well[32];
  int iwell;
  int nwell = 0;
  if (util_file_exists(well_file)) {
    int nread;
    fileH = fopen(well_file , "r");
    while ( (nread = fscanf(fileH , "%s" , well)) == 1)
      nwell++;
    well_list = util_alloc_string_list(nwell , 32);
    rewind(fileH);

    iwell = 0;
    while ( (nread = fscanf(fileH , "%s" , well_list[iwell])) == 1)
      iwell++;
    fclose(fileH);
  }
  *_nwell = nwell;
  return well_list;
}


void ecl_diag_avg_production_interactive(const char *out_path , const char * eclbase_dir , const char * eclbase_avg , const char * eclbase_std , bool fmt_file , bool unified) {
#define defvar_N 3
  const char *defvar_list[defvar_N] = {"WOPR" , "WGOR" , "WWCT"};
  char **well_list;
  /*char **var_list;*/
  int nwell,nvar;
  
  util_make_path(out_path);
  well_list = fread_alloc_wells("wells.dat" , &nwell);
  if (nwell == 0) {
    fprintf(stderr,"Could not find well file: wells.dat - returning \n");
    return;
  }
  /*var_list = defvar_list;*/
  nvar     = defvar_N;
  ecl_diag_avg_production(out_path , GLOBAL_HISTORY , eclbase_dir , eclbase_avg , eclbase_std , nwell , (const char **) well_list , nvar , defvar_list , fmt_file , unified , ENDIAN_CONVERT);
}
	  


void sched_inter_hist_get__(const int *report_step , const char * _well , const int * well_len, const char * _var , const int * var_len , double * value) {
  char *var  = util_alloc_cstring(_var  , var_len);
  char *well = util_alloc_cstring(_well , well_len);

  *value = history_get(GLOBAL_HISTORY , *report_step , well , var);
  
  free(var);
  free(well);
}




void sched_inter_get_report_date__(const int * report_step , int * day, int * month , int * year) {
  time_t t = history_get_report_date(GLOBAL_HISTORY , (*report_step) - 1);
  util_set_date_values(t , day , month , year);
  
}
  







