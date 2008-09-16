#include <stdlib.h>
#include <string.h>
#include <list.h>
#include <hash.h>
#include <time.h>
#include <util.h>
#include <sched_util.h>
#include <sched_kw_dates.h>
#include <history.h>
#include "date_node.h"

struct sched_kw_dates_struct {
  int       *next_date_ptr;
  list_type *date_list;
};


/*****************************************************************/


void sched_kw_dates_add_line(sched_kw_dates_type *kw , const time_t * start_date , const char *line , const hash_type *month_hash, bool TSTEP) {
  date_node_type * date_node = date_node_alloc_from_DATES_line(start_date , *kw->next_date_ptr , line , month_hash);
  list_append_list_owned_ref(kw->date_list , date_node , date_node_free__);
  (*kw->next_date_ptr) += 1;
}



void sched_kw_dates_fprintf(const sched_kw_dates_type *kw , FILE *stream , int last_date_nr , time_t last_time , bool *stop) {
  fprintf(stream,"DATES\n");
  {
    list_node_type *date_node = list_get_head(kw->date_list);
    while (date_node != NULL) {
      const date_node_type * date = list_node_value_ptr(date_node);
      date_node_fprintf(date , stream , last_date_nr , last_time , stop);
      if (*stop) 
	date_node = NULL;
      else
	date_node = list_node_get_next(date_node);
    }
    fprintf(stream , "/\n\n");
  }
}



void sched_kw_dates_get_report_step(const sched_kw_dates_type *kw  , time_t t , int * report_step) {
  list_node_type *date_node = list_get_head(kw->date_list);
  while (date_node != NULL) {
    const date_node_type * date = list_node_value_ptr(date_node);
    date_node_get_report_step(date , t , report_step);
    date_node = list_node_get_next(date_node);
  }
}


void sched_kw_dates_get_time_t(const sched_kw_dates_type *kw  , int report_step , time_t *t) {
  list_node_type *date_node = list_get_head(kw->date_list);
  while (date_node != NULL) {
    const date_node_type * date = list_node_value_ptr(date_node);
    date_node_get_time_t(date , report_step , t);
    date_node = list_node_get_next(date_node);
  }
}



sched_kw_dates_type * sched_kw_dates_alloc(int *next_date_ptr , const time_t * start_date){
  sched_kw_dates_type *dates = malloc(sizeof *dates);
  dates->date_list     = list_alloc();
  dates->next_date_ptr = next_date_ptr;
  return dates;
}

void sched_kw_dates_free(sched_kw_dates_type * kw) {
  list_free(kw->date_list);
  free(kw);
}



void sched_kw_dates_fwrite(const sched_kw_dates_type *kw , FILE *stream) {
  {
    int date_lines = list_get_size(kw->date_list);
    util_fwrite(&date_lines , sizeof date_lines , 1, stream , __func__);
  }
  {
    list_node_type *date_node = list_get_head(kw->date_list);
    while (date_node != NULL) {
      const date_node_type * date = list_node_value_ptr(date_node);
      date_node_fwrite(date ,  stream);
      date_node = list_node_get_next(date_node);
    }
  }
}



sched_kw_dates_type * sched_kw_dates_fread_alloc(int * next_date_ptr , const time_t * start_date , int last_date_nr ,time_t last_time , FILE *stream, bool *stop) {
  int lines , line_nr;
  sched_kw_dates_type *kw = sched_kw_dates_alloc(next_date_ptr , start_date) ;
  util_fread(&lines       , sizeof lines       , 1 , stream , __func__);
  line_nr = 0;
  while (!(*stop) && (line_nr < lines)) {
    date_node_type * date_node = date_node_fread_alloc(start_date , last_date_nr , last_time ,  stream , stop);
    list_append_list_owned_ref(kw->date_list , date_node , date_node_free__);
    line_nr++;
  } 
  return kw;
}


void sched_kw_dates_iterate_current(const sched_kw_dates_type * kw , date_node_type **current_date) {
  (*current_date)     = list_node_value_ptr(list_get_tail(kw->date_list));
}



void sched_kw_dates_fprintf_days_dat(const sched_kw_dates_type * kw , FILE *stream) {
  list_node_type *date_node = list_get_head(kw->date_list);
  while (date_node != NULL) {
    const date_node_type * date = list_node_value_ptr(date_node);
    date_node_fprintf_days_line(date ,  stream);
    date_node = list_node_get_next(date_node);
  }
}


void sched_kw_dates_make_history(const sched_kw_dates_type * kw , history_type * hist) {
  list_node_type *date_node = list_get_head(kw->date_list);
  while (date_node != NULL) {
    const date_node_type * date = list_node_value_ptr(date_node);
    history_add_date(hist , date);
    date_node = list_node_get_next(date_node);
  }
}


int sched_kw_dates_get_size(const sched_kw_dates_type * kw)
{
  return list_get_size(kw->date_list); 
}
