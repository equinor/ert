#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <hash.h>
#include <list.h>
#include <util.h>
#include <time.h>
#include <ecl_kw.h>
#include <sched_util.h>
#include <sched_kw_compdat.h>
#include <sched_kw_dates.h>
#include <sched_kw.h>
#include <sched_file.h>
#include <history.h>


struct sched_file_struct {
  hash_type  *month_hash;
  hash_type  *fixed_record_kw;
  hash_type  *kw_types;
  list_type  *kw_list;
  int         next_date_nr;
  double      acc_days;
  bool        compdat_initialized;
  int        *dims;
  time_t      start_date;
};




/*
  start_date[0] = day
  start_date[1] = month (1-12)
  start_date[2] = year
*/


int sched_file_get_volume(const sched_file_type *s) {
  return s->dims[0] * s->dims[1] * s->dims[2];
}


list_type * sched_file_get_kw_list(const sched_file_type * s) { return s->kw_list; }

time_t sched_file_get_start_date(const sched_file_type * s) { return s->start_date; }

void sched_file_set_start_date(sched_file_type * s , const int * start_date) {
  if (start_date == NULL)
    s->start_date = 0;
  else 
    s->start_date = util_make_time1(start_date[0] , start_date[1] , start_date[2]);
}



sched_file_type * sched_file_alloc(const int *start_date) {
  sched_file_type * sched_file = malloc(sizeof *sched_file);
  {
    hash_type *month_hash = hash_alloc(48);
    hash_insert_int(month_hash , "JAN" , 0);
    hash_insert_int(month_hash , "FEB" , 1);
    hash_insert_int(month_hash , "MAR" , 2);
    hash_insert_int(month_hash , "APR" , 3);
    hash_insert_int(month_hash , "MAY" , 4);
    hash_insert_int(month_hash , "JUN" , 5);
    hash_insert_int(month_hash , "JUL" , 6);
    hash_insert_int(month_hash , "AUG" , 7);
    hash_insert_int(month_hash , "SEP" , 8);
    hash_insert_int(month_hash , "OCT" , 9);
    hash_insert_int(month_hash , "NOV" ,10);
    hash_insert_int(month_hash , "DEC" ,11);
    
    hash_insert_int(month_hash , "jan" , 0);
    hash_insert_int(month_hash , "feb" , 1);
    hash_insert_int(month_hash , "mar" , 2);
    hash_insert_int(month_hash , "apr" , 3);
    hash_insert_int(month_hash , "may" , 4);
    hash_insert_int(month_hash , "jun" , 5);
    hash_insert_int(month_hash , "jul" , 6);
    hash_insert_int(month_hash , "aug" , 7);
    hash_insert_int(month_hash , "sep" , 8);
    hash_insert_int(month_hash , "oct" , 9);
    hash_insert_int(month_hash , "nov" ,10);
    hash_insert_int(month_hash , "dec" ,11);

    hash_insert_int(month_hash , "Jan" , 0);
    hash_insert_int(month_hash , "Feb" , 1);
    hash_insert_int(month_hash , "Mar" , 2);
    hash_insert_int(month_hash , "Apr" , 3);
    hash_insert_int(month_hash , "May" , 4);
    hash_insert_int(month_hash , "Jun" , 5);
    hash_insert_int(month_hash , "Jul" , 6);
    hash_insert_int(month_hash , "Aug" , 7);
    hash_insert_int(month_hash , "Sep" , 8);
    hash_insert_int(month_hash , "Oct" , 9);
    hash_insert_int(month_hash , "Nov" ,10);
    hash_insert_int(month_hash , "Dec" ,11);
    sched_file->month_hash = month_hash;
  }

  {
    hash_type * fixed_record_kw = hash_alloc(10);
    hash_insert_int(fixed_record_kw , "INCLUDE"  , 1);
    hash_insert_int(fixed_record_kw , "RPTSCHED" , 1);
    hash_insert_int(fixed_record_kw , "DRSDT"    , 1);
    hash_insert_int(fixed_record_kw , "SKIPREST" , 0);
    hash_insert_int(fixed_record_kw , "RPTRST"   , 1);
    hash_insert_int(fixed_record_kw , "TSTEP"    , 1);
    hash_insert_int(fixed_record_kw , "TUNING"   , 3);
    sched_file->fixed_record_kw = fixed_record_kw;
  }
  
  {
    hash_type * kw_types = hash_alloc(10);
    hash_insert_int(kw_types , "DATES"    , DATES);
    hash_insert_int(kw_types , "WCONHIST" , WCONHIST);
    hash_insert_int(kw_types , "COMPDAT"  , COMPDAT);
    hash_insert_int(kw_types , "TSTEP"    , TSTEP);
    sched_file->kw_types = kw_types;
  }
  
  sched_file->compdat_initialized = false;
  sched_file->next_date_nr 	  = 1; /* One based or zero based counting... ?? */
  sched_file->acc_days            = 0;
  sched_file->kw_list      	  = list_alloc();
  sched_file->dims                = malloc(3 * sizeof sched_file->dims);
  sched_file_set_start_date(sched_file , start_date);
  return sched_file;
}



void sched_file_add_kw__(sched_file_type *sched_file , const sched_kw_type *kw) {
  list_append_list_owned_ref(sched_file->kw_list , kw , sched_kw_free__); 
}



sched_kw_type * sched_file_add_kw(sched_file_type *sched_file , const char *kw_name) {
  sched_type_enum type;
  bool            fixed_record_kw = false;
  sched_kw_type   *kw;

  if (hash_has_key(sched_file->kw_types , kw_name)) 
    type = hash_get_int(sched_file->kw_types , kw_name);
  else {
    type = UNTYPED;
    if (hash_has_key(sched_file->fixed_record_kw , kw_name))
      fixed_record_kw = true;
    else
      fixed_record_kw = false;
  }
  kw = sched_kw_alloc(kw_name , type , fixed_record_kw , &sched_file->next_date_nr , &sched_file->acc_days , &sched_file->start_date);
  sched_file_add_kw__(sched_file , kw);
  return kw;
}
  


void sched_file_free(sched_file_type *sched_file) {
  list_free(sched_file->kw_list);
  hash_free(sched_file->month_hash);
  hash_free(sched_file->fixed_record_kw);
  hash_free(sched_file->kw_types);
  free(sched_file->dims);
  free(sched_file);
}




void sched_file_parse(sched_file_type * sched_file , const char * filename) {
  int             lines , linenr;
  char          **line_list;
  sched_kw_type  *active_kw;
  int             record_nr , record_size;
  bool            cont;
  
  sched_util_parse_file(filename , &lines , &line_list);
  linenr      = 0;
  active_kw   = NULL;
  cont        = true;
  record_nr   = 0;
  record_size     = -1;
  do {
    const char *line = line_list[linenr];
    if (strncmp(line , "END" , 3) == 0) {
      fprintf(stderr,"%s: Warning: found END statement in file:%s before actual file \n",__func__ , filename);
      fprintf(stderr,"end reached, a risk for premature ending in the parsing of the schedule file.\n");
      cont = false;
    } else {
      if (active_kw == NULL) {
	const char * kw_name = line;
	record_nr            = 0;
	{
	  int index = 0;
	  while (index < strlen(line)) {
	    if (kw_name[index] == ' ') {
	      fprintf(stderr,"%s: Hmmmm - this looks suspicious line: \"%s\" should be a pure keyword, without data - aborting.\n",__func__ , kw_name);
	      abort();
	    }
	    index++;
	  }
	}
	if (hash_has_key(sched_file->fixed_record_kw , kw_name))
	  record_size = hash_get_int(sched_file->fixed_record_kw , kw_name);
	else
	  /*
	    Variable lengt keyword like e.g. COMPDAT.
	  */
	  record_size = -1;


	active_kw = sched_file_add_kw(sched_file , kw_name);
	if (record_size == 0) /* The keyword is already complete ... */
	  active_kw = NULL;
      } else {
	bool complete = true;
	
	if (line[0] == '/') 
	  active_kw = NULL;
	else 
	  sched_kw_add_line(active_kw , line , &sched_file->start_date , sched_file->month_hash , &complete);
	/*
	  The complete variable signals that the current record is complete - i.e. the line terminated with a slash ...

	  Not quite sure how well it handels continuation without a slash ??
	*/
	
	if (record_size >= 1 && complete) {
	  record_nr++;
	  if (record_nr == record_size)
	    active_kw = NULL;
	}
      }
      linenr++;
      if (linenr == (lines - 1)) {
	if (strncmp(line_list[linenr] , "END" , 3) == 0)
	  cont = false;
	else {
	  fprintf(stderr,"%s: Error (internal ?) when parsing %s : not END on last line - aborting \n" , __func__ , filename);
	  abort();
	}
      }
    }
  } while (cont);
  util_free_string_list(line_list , lines);
}


void sched_file_fprintf(const sched_file_type * sched_file , int last_date_nr , time_t last_time , double last_day , const char * file) {
  FILE *stream = fopen(file , "w");
  list_node_type *list_node = list_get_head(sched_file->kw_list);
  bool stop                 = false;
  
  while (list_node != NULL) {
    const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
    sched_kw_fprintf(sched_kw , last_date_nr , last_time , last_day , stream , &stop);
    if (stop)
      list_node = NULL;
    else
      list_node = list_node_get_next(list_node);
  }
  fprintf(stream , "END\n");

  fclose(stream);
}



void sched_file_init_conn_factor(sched_file_type * sched_file , const char * init_file , bool endian_flip , const int * index_map) {
  bool OK;
  ecl_kw_type * permx_kw , *ihead_kw , *permz_kw;
  bool fmt_file        = util_fmt_bit8(init_file );
  fortio_type * fortio = fortio_open(init_file , "r" , endian_flip);

  ecl_kw_fseek_kw("INTEHEAD" , fmt_file , true , true , fortio);
  ihead_kw = ecl_kw_fread_alloc(fortio , fmt_file);

  ecl_kw_fseek_kw("PERMX" , fmt_file , true , true , fortio);
  permx_kw = ecl_kw_fread_alloc(fortio , fmt_file );

  ecl_kw_fseek_kw("PERMZ" , fmt_file , true , true , fortio);
  permz_kw = ecl_kw_fread_alloc(fortio , fmt_file );
  
  fortio_close(fortio);

  {
    int * tmp  = ecl_kw_get_data_ref(ihead_kw);
    memcpy(sched_file->dims , &tmp[8] , 3 * sizeof sched_file->dims);
  }
  
  OK = true;
  {
    list_node_type *list_node = list_get_head(sched_file->kw_list);
    while (list_node != NULL) {
      const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
      if (sched_kw_get_type(sched_kw) == COMPDAT) 
	sched_kw_compdat_init_conn_factor(sched_kw_get_data_ref(sched_kw) , permx_kw , permz_kw , sched_file->dims, index_map , &OK);
      
      list_node = list_node_get_next(list_node);
    }
  }
  if (OK)
    sched_file->compdat_initialized = true;
  ecl_kw_free(permx_kw);
  ecl_kw_free(ihead_kw);
  ecl_kw_free(permz_kw);
}



void sched_file_set_conn_factor(sched_file_type * sched_file , const float * permx , const float * permz , const int * index_map) {
  if (sched_file->compdat_initialized) {
    list_node_type *list_node = list_get_head(sched_file->kw_list);
    while (list_node != NULL) {
      const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
      if (sched_kw_get_type(sched_kw) == COMPDAT) 
	sched_kw_compdat_set_conn_factor(sched_kw_get_data_ref(sched_kw) , permx , permz , sched_file->dims , index_map);
      
      list_node = list_node_get_next(list_node);
    }
  } else {
    fprintf(stderr, "%s: must call sched_file_init_conn_factor first - aborting \n",__func__);
    abort();
  }
}


void sched_file_fwrite(const sched_file_type * sched_file , FILE * stream) {
  {
    int len = list_get_size(sched_file->kw_list);
    util_fwrite(&len , sizeof len , 1 , stream , __func__);
  }

  util_fwrite(&sched_file->compdat_initialized , sizeof sched_file->compdat_initialized , 1 , stream , __func__);
  util_fwrite(sched_file->dims                 , sizeof sched_file->dims       	        , 3 , stream , __func__); 
  util_fwrite(&sched_file->start_date          , sizeof sched_file->start_date 	        , 1 , stream , __func__);

  {
    list_node_type *list_node = list_get_head(sched_file->kw_list);
    while (list_node != NULL) {
      const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
      sched_kw_fwrite(sched_kw , stream);
      list_node = list_node_get_next(list_node);
    }
  }
}


sched_file_type * sched_file_fread_alloc(FILE *stream, int last_date_nr , time_t last_time , double last_day) {
  bool cont , at_eof , stop;
  int len,kw_nr;
  sched_file_type * sched_file;
  
  sched_file = sched_file_alloc(NULL);
  util_fread(&len                             , sizeof len                             , 1 , stream , __func__); 
  util_fread(&sched_file->compdat_initialized , sizeof sched_file->compdat_initialized , 1 , stream , __func__);
  util_fread(sched_file->dims                 , sizeof sched_file->dims                , 3 , stream , __func__); 
  util_fread(&sched_file->start_date          , sizeof sched_file->start_date          , 1 , stream , __func__); 
  
  at_eof = false;
  stop   = false;
  cont   = true;
  kw_nr  = 0;
  while (cont) {
    sched_kw_type *kw = sched_kw_fread_alloc(&sched_file->next_date_nr , &sched_file->acc_days , &sched_file->start_date , last_date_nr , last_time , last_day ,stream , &at_eof, &stop);
    sched_file_add_kw__(sched_file , kw);
    kw_nr += 1;
    if (at_eof || stop || kw_nr == len)
      cont = false;
  }
  
  if (kw_nr < len && !stop) 
    fprintf(stderr,"%s: Warning premature end in schedule dump file read %d/%d keywords.\n",__func__ , kw_nr , len);
  
  return sched_file;
}



void sched_file_fprintf_rates(const sched_file_type * sched_file , const char * obs_path, const char * obs_file) {
  date_node_type *current_date = NULL;
  list_node_type *list_node = list_get_head(sched_file->kw_list);
  util_make_path(obs_path);
  printf("Writing observations: "); fflush(stdout);
  while (list_node != NULL) {
    const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
    sched_kw_fprintf_rates(sched_kw , obs_path , obs_file , &current_date);
    list_node = list_node_get_next(list_node);
  }
  printf("complete \n");
}



void sched_file_fprintf_days_dat(const sched_file_type *s , const char *days_file) {
  list_node_type *list_node = list_get_head(s->kw_list);
  FILE * stream = util_fopen(days_file , "w");
  sched_util_fprintf_days_line(0 , s->start_date , s->start_date , stream);
  
  while (list_node != NULL) {
    const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
    sched_kw_fprintf_days_dat(sched_kw , stream);
    list_node = list_node_get_next(list_node);
  }
  
  fclose(stream);
}



int sched_file_time_t_to_report_step(const sched_file_type * s , time_t t , int * _status) {
  int report_step = -1;
  int status = -1;
  if (difftime(t , s->start_date)  >= 0) {
    list_node_type *list_node = list_get_head(s->kw_list);
    while (list_node != NULL) {
      const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
      sched_kw_get_report_step(sched_kw , t , &report_step);

      if (report_step != -1) {
	status = 0;
	list_node = NULL;
      } else
	list_node = list_node_get_next(list_node);
    } 
    if (report_step == -1) 
      status = 1;
  }
  /*
   *_status = status;
   */
  
  if (_status != NULL)
    *_status = status;

  if (status != 0) {
    if (_status == NULL) {
      int year , mday , mon;
      util_set_date_values(t , &mday , &mon , &year);
      if (status < 0)
	fprintf(stderr,"%s: *Warning* time %02d/%02d/%4d is before simulation start - report step = -1 is assigned. \n",__func__ , mday , mon , year);
      else
	fprintf(stderr,"%s: *Warning* time %02d/%02d/%4d is after simulation end - report step = -1 is assigned. \n",__func__ , mday , mon , year);
    }
  }
  return report_step;
}


int sched_file_int3_to_report_step(const sched_file_type * s , int mday, int mon , int year , int * status) {
  time_t t         = util_make_time1(mday , mon , year);
  return sched_file_time_t_to_report_step(s , t , status);
}


int sched_file_DATES_to_report_step(const sched_file_type * s , const char * DATES_line , int * status) {
  return sched_file_time_t_to_report_step(s , date_node_parse_DATES_line(DATES_line , s->month_hash) , status);
}


