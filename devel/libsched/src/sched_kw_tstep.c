#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <double_vector.h>
#include <sched_util.h>
#include <util.h>
#include <sched_kw_tstep.h>
#include <sched_macros.h>



struct sched_kw_tstep_struct {
  double_vector_type * tstep_list;
}; 



/*****************************************************************/


static void sched_kw_tstep_add_tstep( sched_kw_tstep_type * kw, double tstep ) {
  double_vector_append( kw->tstep_list , tstep );
}


static void sched_kw_tstep_add_tstep_string( sched_kw_tstep_type * kw, const char * tstep_string) {
  double tstep;
  if (util_sscanf_double( tstep_string , &tstep ))
    sched_kw_tstep_add_tstep(kw , tstep );
  else
    util_abort("%s: failed to parse:%s as a floating point number \n",__func__ , tstep_string);
}




static void sched_kw_tstep_add_line(sched_kw_tstep_type *kw , const char *line , bool *complete) {
  char **tstep_list;
  int i , steps;
  sched_util_parse_line(line , &steps , &tstep_list , 1 , complete);
  for (i=0; i  < steps; i++) 
    sched_kw_tstep_add_tstep_string( kw , tstep_list[i]);

  {
    int len = strlen(tstep_list[steps-1]);
    if(tstep_list[steps-1][len-1] == '/')
      *complete = true;
  }
  util_free_stringlist(tstep_list , steps);
}



static sched_kw_tstep_type * sched_kw_tstep_alloc(){
  sched_kw_tstep_type *tstep = util_malloc(sizeof * tstep , __func__ );
  tstep->tstep_list          = double_vector_alloc(0 , 0);
  return tstep;
}



/*****************************************************************/

sched_kw_tstep_type * sched_kw_tstep_token_alloc(const stringlist_type * tokens , int * token_index ) {
  sched_kw_tstep_type * kw = sched_kw_tstep_alloc();
  int eokw                    = false;
  do {
    stringlist_type * line_tokens = sched_util_alloc_line_tokens( tokens , false , 0 , token_index );
    if (line_tokens == NULL)
      eokw = true;
    else {
      int i;
      for (i=0; i < stringlist_get_size( line_tokens ); i++)
        sched_kw_tstep_add_tstep_string( kw , stringlist_iget( line_tokens , i ));
      stringlist_free( line_tokens );
    } 
  } while (!eokw);
  return kw;
}



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
    int i;
    for (i=0; i < double_vector_size( kw->tstep_list ); i++)
      fprintf(stream, "%7.3f", double_vector_iget( kw->tstep_list , i));
  }
  fprintf(stream , " /\n\n");
}



void sched_kw_tstep_free(sched_kw_tstep_type * kw) {
  double_vector_free(kw->tstep_list);
  free(kw);
}



void sched_kw_tstep_fwrite(const sched_kw_tstep_type *kw , FILE *stream) {
  //{
  //  int tstep_lines = list_get_size(kw->tstep_list);
  //  util_fwrite(&tstep_lines , sizeof tstep_lines , 1, stream , __func__);
  //}
  //{
  //  list_node_type *list_node = list_get_head(kw->tstep_list);
  //  while (list_node != NULL) {
  //    const double * step = list_node_value_ptr(list_node);
  //    util_fwrite(step, sizeof *step, 1, stream, __func__);
  //    list_node = list_node_get_next(list_node);
  //  }
  //}
}



sched_kw_tstep_type * sched_kw_tstep_fread_alloc(FILE * stream) {
  //int lines , line_nr;
  //sched_kw_tstep_type *kw = sched_kw_tstep_alloc();
  //util_fread(&lines, sizeof lines, 1, stream, __func__);
  //line_nr = 0;
  //while (line_nr < lines) {
  //  double * step = util_malloc(sizeof * step, __func__);
  //  util_fread(step, sizeof * step, 1, stream, __func__);
  //  list_append_list_owned_ref(kw->tstep_list , step, free);
  //  line_nr++;
  //} 
  //return kw;
  return NULL;
}



int sched_kw_tstep_get_size(const sched_kw_tstep_type * kw)
{
  return double_vector_size(kw->tstep_list);
}



sched_kw_tstep_type * sched_kw_tstep_alloc_from_double(double step)
{
  sched_kw_tstep_type * kw = sched_kw_tstep_alloc();
  double_vector_append( kw->tstep_list , step );
  return kw;
}



double sched_kw_tstep_iget_step(const sched_kw_tstep_type * kw, int i)
{
  return double_vector_iget( kw->tstep_list , i );
}



double sched_kw_tstep_get_step(const sched_kw_tstep_type * kw)
{
  if(sched_kw_tstep_get_size(kw) > 1)
  {
    util_abort("%s: Internal error - must use sched_kw_tstep_iget_step instead - aborting\n", __func__);
  }

  return sched_kw_tstep_iget_step(kw, 0);
}



time_t sched_kw_tstep_get_new_time(const sched_kw_tstep_type *kw, time_t curr_time)
{
  double step_days = sched_kw_tstep_iget_step(kw , 0);
  time_t new_time  = curr_time;
  util_inplace_forward_days(&new_time, step_days);
  return new_time;
}


/*****************************************************************/

KW_IMPL(tstep)


