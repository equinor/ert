#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <list.h>
#include <util.h>
#include <sched_util.h>
#include <sched_kw_tstep.h>
#include <sched_macros.h>



struct sched_kw_tstep_struct {
  list_type    * tstep_list; /* A list of doubles. */
}; 



/*****************************************************************/



static void sched_kw_tstep_add_line(sched_kw_tstep_type *kw , const char *line , bool *complete) {
  char **tstep_list;
  int i , steps;
  sched_util_parse_line(line , &steps , &tstep_list , 1 , complete);
  for (i=0; i  < steps; i++) {
    double * step = util_malloc(sizeof * step, __func__);
    *step = atof(tstep_list[i]);
    list_append_list_owned_ref(kw->tstep_list , step , free);
  }
  {
    int len = strlen(tstep_list[steps-1]);
    if(tstep_list[steps-1][len-1] == '/')
      *complete = true;
  }
  
  sched_util_free_token_list(steps , tstep_list);
}



static sched_kw_tstep_type * sched_kw_tstep_alloc(){
  sched_kw_tstep_type *tstep = malloc(sizeof *tstep);
  tstep->tstep_list     = list_alloc();
  return tstep;
}



/*****************************************************************/



sched_kw_tstep_type * sched_kw_tstep_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_tstep_type * kw = sched_kw_tstep_alloc();

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      free(line);
      break;
    }
    else if(line != NULL)
    {
      sched_kw_tstep_add_line(kw, line, &at_eokw);
      free(line);
    }
  }

  return kw;
}



void sched_kw_tstep_fprintf(const sched_kw_tstep_type *kw , FILE *stream) {
  fprintf(stream,"TSTEP\n  ");
  {
    list_node_type *list_node = list_get_head(kw->tstep_list);
    while (list_node != NULL) {
      const double * step = list_node_value_ptr(list_node);
      fprintf(stream, "%7.3f", *step);
	    list_node = list_node_get_next(list_node);
    }
    fprintf(stream , " /\n\n");
  }
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
    list_node_type *list_node = list_get_head(kw->tstep_list);
    while (list_node != NULL) {
      const double * step = list_node_value_ptr(list_node);
      util_fwrite(step, sizeof *step, 1, stream, __func__);
      list_node = list_node_get_next(list_node);
    }
  }
}



sched_kw_tstep_type * sched_kw_tstep_fread_alloc(FILE * stream) {
  int lines , line_nr;
  sched_kw_tstep_type *kw = sched_kw_tstep_alloc();
  util_fread(&lines, sizeof lines, 1, stream, __func__);
  line_nr = 0;
  while (line_nr < lines) {
    double * step = util_malloc(sizeof * step, __func__);
    util_fread(step, sizeof * step, 1, stream, __func__);
    list_append_list_owned_ref(kw->tstep_list , step, free);
    line_nr++;
  } 
  return kw;
}



int sched_kw_tstep_get_size(const sched_kw_tstep_type * kw)
{
  return list_get_size(kw->tstep_list);
}



sched_kw_tstep_type * sched_kw_tstep_alloc_from_double(double step)
{
  sched_kw_tstep_type * kw = sched_kw_tstep_alloc();
  double * step_stor = util_malloc(sizeof *step_stor, __func__);
  *step_stor = step;
  list_append_list_owned_ref(kw->tstep_list, step_stor, free);
  return kw;
}



double sched_kw_tstep_iget_step(const sched_kw_tstep_type * kw, int i)
{
  double step;
  double * step_stor;

  list_node_type * step_node = list_iget_node(kw->tstep_list, i);
  step_stor = (double *) list_node_value_ptr(step_node);
  step = *step_stor;

  return step;
}


double sched_kw_tstep_get_step(const sched_kw_tstep_type * kw)
{
  if(sched_kw_tstep_get_size(kw) > 1)
  {
    util_abort("%s: Internal error - must use sched_kw_tstep_iget_step instead - aborting\n",
               __func__);
  }

  return sched_kw_tstep_iget_step(kw, 0);
}



time_t sched_kw_tstep_get_new_time(const sched_kw_tstep_type *kw, time_t curr_time)
{
  double step_days = sched_kw_tstep_get_step(kw);
  time_t step_time = (time_t) (60*60*24*step_days);
  return step_time + curr_time;
}


/*****************************************************************/

KW_FSCANF_ALLOC_IMPL(tstep)
KW_FWRITE_IMPL(tstep)
KW_FREAD_ALLOC_IMPL(tstep)
KW_FREE_IMPL(tstep)
KW_FPRINTF_IMPL(tstep)


