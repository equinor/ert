#include <stdlib.h>
#include <list.h>
#include <time.h>
#include <util.h>
#include <sched_util.h>
#include <sched_kw_tstep.h>
#include <date_node.h>


struct sched_kw_tstep_struct {
  const time_t * start_date;
  int          * next_tstep_ptr;
  double       * acc_days_ptr;
  list_type    * tstep_list;
}; 



typedef struct {
  double   acc_days;
  double   step_days;
  int      tstep_nr;
} tstep_node_type;



/*****************************************************************/

static tstep_node_type * tstep_node_alloc(int * tstep_nr , double * acc_days , const char *line) {
  tstep_node_type *tstep_node = malloc(sizeof *tstep_node);


  tstep_node->tstep_nr        = *tstep_nr; 
  tstep_node->step_days       = atof(line);
  tstep_node->acc_days        = *acc_days + tstep_node->step_days;
  
  (*tstep_nr) += 1;
  (*acc_days) += tstep_node->step_days;
  
  return tstep_node;
}


static void tstep_node_free(tstep_node_type *tstep) {
  free(tstep);
}


static void tstep_node_free__(void *__tstep) {
  tstep_node_free((tstep_node_type *) __tstep);
}


static void tstep_node_fprintf(const tstep_node_type * node, FILE *stream , int last_tstep_nr , double last_day , bool *stop) {
  
  fprintf(stream , "%7.3f " , node->step_days);
  if (last_tstep_nr > 0) {
    if (node->tstep_nr >= last_tstep_nr)
      *stop = true;
  }

  if (last_day > 0) {
    if (node->acc_days >= last_day)
      *stop = true;
  }
  
}

static void tstep_node_fwrite(const tstep_node_type * tstep_node , FILE *stream) {
  util_fwrite(&tstep_node->acc_days   , sizeof tstep_node->acc_days   , 1 , stream , __func__);
  util_fwrite(&tstep_node->step_days  , sizeof tstep_node->step_days  , 1 , stream , __func__);
  util_fwrite(&tstep_node->tstep_nr   , sizeof tstep_node->tstep_nr   , 1 , stream , __func__);
}


static tstep_node_type * tstep_node_fread_alloc(int last_tstep_nr , double last_day , FILE *stream, bool *stop) {
  tstep_node_type * node = malloc(sizeof *node); /* SKipping spesific alloc routine - UGGLY */
  util_fread(&node->acc_days    , sizeof node->acc_days    , 1 , stream , __func__);
  util_fread(&node->step_days   , sizeof node->step_days   , 1 , stream , __func__);
  util_fread(&node->tstep_nr    , sizeof node->tstep_nr    , 1 , stream , __func__);

  if (last_tstep_nr > 0) {
    if (node->tstep_nr >= last_tstep_nr)
      *stop = true;
  }
  
  if (last_day > 0) {
    if (node->acc_days >= last_day)
      *stop = true;
  }
  return node;
}

/*****************************************************************/


void sched_kw_tstep_add_line(sched_kw_tstep_type *kw , const char *line , bool *complete) {
  char **tstep_list;
  int i , steps;
  sched_util_parse_line(line , &steps , &tstep_list , 1 , complete);
  for (i=0; i  < steps; i++) {
    tstep_node_type * tstep_node = tstep_node_alloc(kw->next_tstep_ptr , kw->acc_days_ptr , tstep_list[i]);
    list_append_list_owned_ref(kw->tstep_list , tstep_node , tstep_node_free__);
  }
  
  sched_util_free_token_list(steps , tstep_list);
}


void sched_kw_tstep_fprintf(const sched_kw_tstep_type *kw , FILE *stream , int last_tstep_nr , time_t last_time , bool *stop) {
  fprintf(stream,"TSTEP\n  ");
  {
    list_node_type *tstep_node = list_get_head(kw->tstep_list);
    while (tstep_node != NULL) {
      const tstep_node_type * tstep = list_node_value_ptr(tstep_node);
      tstep_node_fprintf(tstep , stream , last_tstep_nr , last_time , stop);
      if (*stop) 
	tstep_node = NULL;
      else
	tstep_node = list_node_get_next(tstep_node);
    }
    fprintf(stream , "/\n\n");
  }
}


sched_kw_tstep_type * sched_kw_tstep_alloc(int *next_tstep_ptr , double *acc_days_ptr , const time_t * start_date){
  sched_kw_tstep_type *tstep = malloc(sizeof *tstep);
  tstep->tstep_list     = list_alloc();
  tstep->next_tstep_ptr = next_tstep_ptr;
  tstep->acc_days_ptr   = acc_days_ptr;
  tstep->start_date     = start_date;
  return tstep;
}


void sched_kw_tstep_free(sched_kw_tstep_type * kw) {
  list_free(kw->tstep_list);
  free(kw);
}



void sched_kw_tstep_fwrite(const sched_kw_tstep_type *kw , FILE *stream) {
  {
    int tstep_lines = list_get_size(kw->tstep_list);
    util_fwrite(&tstep_lines , sizeof tstep_lines , 1, stream , __func__);
  }
  {
    list_node_type *tstep_node = list_get_head(kw->tstep_list);
    while (tstep_node != NULL) {
      const tstep_node_type * tstep = list_node_value_ptr(tstep_node);
      tstep_node_fwrite(tstep ,  stream);
      tstep_node = list_node_get_next(tstep_node);
    }
  }
}



sched_kw_tstep_type * sched_kw_tstep_fread_alloc(int * next_tstep_ptr , double * acc_days_ptr , const time_t * start_date , int last_tstep_nr ,time_t last_time , FILE *stream, bool *stop) {
  int lines , line_nr;
  sched_kw_tstep_type *kw = sched_kw_tstep_alloc(next_tstep_ptr , acc_days_ptr , start_date) ;
  util_fread(&lines       , sizeof lines       , 1 , stream , __func__);
  line_nr = 0;
  while (!(*stop) && (line_nr < lines)) {
    tstep_node_type * tstep_node = tstep_node_fread_alloc(last_tstep_nr , last_time ,  stream , stop);
    list_append_list_owned_ref(kw->tstep_list , tstep_node , tstep_node_free__);
    line_nr++;
  } 
  return kw;
}


void sched_kw_tstep_iterate_current(const sched_kw_tstep_type * kw , int *current_tstep_nr) {
  (*current_tstep_nr) += list_get_size(kw->tstep_list);
}



int sched_kw_tstep_get_size(const sched_kw_tstep_type * kw)
{
  return list_get_size(kw->tstep_list);
}
