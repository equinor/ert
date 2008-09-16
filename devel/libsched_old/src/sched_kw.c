#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <hash.h>
#include <time.h>
#include <util.h>
#include <sched_kw.h>
#include <sched_kw_compdat.h>
#include <sched_kw_wconhist.h>
#include <sched_kw_dates.h>
#include <sched_kw_tstep.h>
#include <sched_kw_welspecs.h>
#include <sched_kw_gruptree.h>
#include <sched_kw_untyped.h>


struct sched_kw_struct {
  sched_type_enum    type;
  void              *data;
};


/*****************************************************************/

void * sched_kw_get_data_ref(const sched_kw_type * kw) {return kw->data; }
sched_type_enum sched_kw_get_type(const sched_kw_type * kw) { return kw->type; }

sched_kw_type * sched_kw_alloc(const char * kw_name , sched_type_enum type, bool one_line_kw, int * next_date_nr , double * acc_days_ptr, const time_t * start_date) {
  
  sched_kw_type *kw = malloc(sizeof *kw);
  kw->type = type;
  
  switch (kw->type) {
  case(COMPDAT):
    kw->data = sched_kw_compdat_alloc();
    break;
  case(WCONHIST):
    kw->data = sched_kw_wconhist_alloc();
    break;
  case(DATES):
    kw->data = sched_kw_dates_alloc(next_date_nr , start_date);
    break;
  case(TSTEP):
    kw->data = sched_kw_tstep_alloc(next_date_nr , acc_days_ptr , start_date);
    break;
  case(WELSPECS):
    kw->data = sched_kw_welspecs_alloc();
    break;
  case(GRUPTREE):
    kw->data = sched_kw_gruptree_alloc();
    break;
  case(UNTYPED):
    kw->data = sched_kw_untyped_alloc(kw_name , one_line_kw);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type - aborting \n",__func__);
    abort();
  }
  return kw;
}



void sched_kw_free(sched_kw_type * kw) {
  switch (kw->type) {
  case(COMPDAT):
    sched_kw_compdat_free(kw->data);
    break;
  case(WCONHIST):
    sched_kw_wconhist_free(kw->data);
    break;
  case(DATES):
    sched_kw_dates_free(kw->data);
    break;
  case(TSTEP):
    sched_kw_tstep_free(kw->data);
    break;
  case(WELSPECS):
    sched_kw_welspecs_free(kw->data);
    break;
  case(GRUPTREE):
    sched_kw_gruptree_free(kw->data);
    break;
  case(UNTYPED):
    sched_kw_untyped_free(kw->data);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type - aborting \n",__func__);
    abort();
  }
  free(kw);
}


void sched_kw_free__(void * void_kw) {
  sched_kw_free( (sched_kw_type *) void_kw);
}



void sched_kw_add_line(sched_kw_type * kw, const char * line, const time_t *start_date , const hash_type *month_hash , bool *complete) {
  switch (kw->type) {
  case(COMPDAT):
    sched_kw_compdat_add_line(kw->data , line);
    break;
  case(WCONHIST):
    sched_kw_wconhist_add_line(kw->data , line);
    break;
  case(DATES):
    sched_kw_dates_add_line(kw->data , start_date , line , month_hash , false);
    break;
  case(TSTEP):
    sched_kw_tstep_add_line(kw->data , line , complete);
    break;
  case(WELSPECS):
    sched_kw_welspecs_add_line(kw->data, line);
    break;
  case(GRUPTREE):
    sched_kw_gruptree_add_line(kw->data, line);
    break;
  case(UNTYPED):
    sched_kw_untyped_add_line(kw->data , line);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type:%d - aborting \n",__func__ , kw->type);
    fprintf(stderr,"%s: current line: \"%s\" \n",__func__ , line);
    abort();
  }
}




void sched_kw_fprintf(const sched_kw_type * kw , int last_date_nr , time_t last_time , double last_day , FILE *stream , bool *stop) {
  switch (kw->type) {
  case(COMPDAT):
    sched_kw_compdat_fprintf(kw->data , stream);
    break;
  case(WCONHIST):
    sched_kw_wconhist_fprintf(kw->data , stream);
    break;
  case(DATES):
    sched_kw_dates_fprintf(kw->data , stream, last_date_nr , last_time , stop);
    break;
  case(TSTEP):
    sched_kw_tstep_fprintf(kw->data , stream, last_date_nr , last_day , stop);
    break;
  case(WELSPECS):
    sched_kw_welspecs_fprintf(kw->data, stream);
    break;
  case(GRUPTREE):
    sched_kw_gruptree_fprintf(kw->data, stream);
    break;
  case(UNTYPED):
    sched_kw_untyped_fprintf(kw->data , stream);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type - aborting \n",__func__);
    abort();
  }
}


void sched_kw_fwrite(const sched_kw_type *kw , FILE *stream) {
  util_fwrite(&kw->type , sizeof kw->type , 1 , stream , __func__);
  switch (kw->type) {
  case(COMPDAT):
    sched_kw_compdat_fwrite(kw->data , stream);
    break;
  case(WCONHIST):
    sched_kw_wconhist_fwrite(kw->data , stream);
    break;
  case(DATES):
    sched_kw_dates_fwrite(kw->data , stream);
    break;
  case(TSTEP):
    sched_kw_tstep_fwrite(kw->data , stream);
    break;
  case(WELSPECS):
    sched_kw_welspecs_fwrite(kw->data, stream);
    break;
  case(GRUPTREE):
    sched_kw_gruptree_fwrite(kw->data, stream);
    break;
  case(UNTYPED):
    sched_kw_untyped_fwrite(kw->data , stream);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type=%d - aborting \n",__func__ , kw->type);
    abort();
  }
}


sched_kw_type * sched_kw_fread_alloc(int *next_date_nr , double *acc_days_ptr , const time_t *start_date , int last_date_nr , time_t last_time , double last_day , FILE *stream , bool *at_eof, bool *stop) {
  sched_kw_type * kw = malloc(sizeof *kw);
  util_fread(&kw->type  , sizeof kw->type , 1 , stream , __func__);

  switch (kw->type) {
  case(COMPDAT):
    kw->data = sched_kw_compdat_fread_alloc(stream);
    break;
  case(WCONHIST):
    kw->data = sched_kw_wconhist_fread_alloc(stream);
    break;
  case(DATES):
    kw->data = sched_kw_dates_fread_alloc(next_date_nr, start_date , last_date_nr , last_time , stream , stop);
    break;
  case(TSTEP):
    kw->data = sched_kw_tstep_fread_alloc(next_date_nr, acc_days_ptr , start_date , last_date_nr , last_day , stream , stop);
    break;
  case(WELSPECS):
    kw->data = sched_kw_welspecs_fread_alloc(stream);
    break;
  case(GRUPTREE):
    kw->data = sched_kw_gruptree_fread_alloc(stream);
    break;
  case(UNTYPED):
    kw->data = sched_kw_untyped_fread_alloc(stream);
    break;
  default:
    fprintf(stderr,"%s: internal error - unknown value for kw->type - aborting \n",__func__);
    abort();
  }
  
  {
    char next_c = fgetc(stream);
    if (feof(stream)) {
      *at_eof = true;
    }else {
      *at_eof = false;
      ungetc(next_c , stream);
    }
  }
  
  return kw;
}


void sched_kw_fprintf_rates(const sched_kw_type * kw , const char *obs_path , const char * obs_file , date_node_type **current_date) {
  if (kw->type == WCONHIST) 
    sched_kw_wconhist_fprintf_rates(kw->data , obs_path , obs_file , date_node_get_date_nr(*current_date) + 1);  /* Current_date is the last, i.e. previous, date. */
  else if (kw->type == DATES) {
    sched_kw_dates_iterate_current(kw->data , current_date);
    date_node_fprintf_rate_date(*current_date , obs_path , "DATE");
  }
}



/*
  TODO

  This will create a bug if exposed to TSTEP.
*/
void sched_kw_fprintf_days_dat(const sched_kw_type * kw , FILE *stream) {
  if (kw->type == DATES) 
    sched_kw_dates_fprintf_days_dat(kw->data , stream);
}



/*
  TODO

  This will create a bug if exposed to TSTEP.
*/
void sched_kw_get_report_step(const sched_kw_type * kw , time_t t , int * report_step) {
  if (kw->type == DATES) 
    sched_kw_dates_get_report_step(kw->data , t , report_step);
}



/**
   This function takes a report_step as input, and sets the time_t
   instance accordingly. If the current sched_kw instance is not of
   DATES type nothing happens to the time_t instance.

   I.e. the calling scope must monitor the value of t to check whether
   it has been updated.

   TODO

   This will create a bug if exposed to TSTEP.
*/
void sched_kw_get_time_t(const sched_kw_type * kw , int report_step, time_t *t) {
  if (kw->type == DATES) 
    sched_kw_dates_get_time_t(kw->data ,  report_step , t);
}



/*
   TODO 


   This will create a bug if exposed to TSTEP.
*/
void sched_kw_make_history(const sched_kw_type * kw , history_type * history, date_node_type **current_date) {
  if (kw->type == DATES) {
    sched_kw_dates_make_history(kw->data  , history);
    sched_kw_dates_iterate_current(kw->data , current_date);
  }
  else if (kw->type == WCONHIST) 
    sched_kw_wconhist_make_history(kw->data , date_node_get_date_nr(*current_date) + 1 , history);
}



/*****************************************************************/















