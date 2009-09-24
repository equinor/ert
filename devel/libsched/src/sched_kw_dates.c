#include <stdlib.h>
#include <string.h>
#include <time_t_vector.h>
#include <time.h>
#include <util.h>
#include <sched_util.h>
#include <sched_kw_dates.h>
#include <stringlist.h>


struct sched_kw_dates_struct {
  time_t_vector_type  * date_list;
};



/*****************************************************************/



static sched_kw_dates_type * sched_kw_dates_alloc()
{
  sched_kw_dates_type *dates = util_malloc(sizeof *dates , __func__);
  dates->date_list           = time_t_vector_alloc(0 , -1);
  return dates;
}



static const char * get_month_string_from_int(int month_nr)
{
  switch(month_nr)
  {
    case(1):  return "JAN";
    case(2):  return "FEB";
    case(3):  return "MAR";
    case(4):  return "APR";
    case(5):  return "MAY";
    case(6):  return "JUN";
    case(7):  return "JUL";
    case(8):  return "AUG";
    case(9):  return "SEP";
    case(10): return "OCT";
    case(11): return "NOV";
    case(12): return "DEC";
    default:
      util_abort("%s: Internal error - %i is not a month nr.\n",__func__,month_nr);
      return "ERR\0";
  }

}



static time_t parse_time_t_from_dates_line(const char * line)
{
  int mday , month , year, tokens;
  char **token_list;
  time_t time = -1;

  sched_util_parse_line(line ,&tokens ,&token_list , 3, NULL);
  month = util_get_month_nr(token_list[1]);

  if (util_sscanf_int(token_list[0] , &mday) && util_sscanf_int(token_list[2] , &year))
    time = util_make_date(mday , month , year);
  else 
    util_abort("%s: fatal error when extracting date from:%s \n", __func__, line);
  
  util_free_stringlist(token_list , tokens);
  
  return time;
}



static void time_t_to_dates_line_fprintf(time_t date, FILE * stream)
{
  int day, month, year;
  util_set_date_values(date, &day, &month, &year);
  fprintf(stream , "  %d \'%s\' %4d  /  \n" , day, get_month_string_from_int(month), year );
}



static void sched_kw_dates_add_line(sched_kw_dates_type *kw, const char *line) {
  time_t date = parse_time_t_from_dates_line(line);
  time_t_vector_append(kw->date_list , date);
}



/*****************************************************************/


sched_kw_dates_type * sched_kw_dates_token_alloc(const stringlist_type * tokens , int * __token_index ) {
  
}


sched_kw_dates_type  * sched_kw_dates_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_dates_type * kw = sched_kw_dates_alloc();

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before DATES was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_dates_add_line(kw, line);
      free(line);
    }
  }
  return kw;
}



void sched_kw_dates_fprintf(const sched_kw_dates_type *kw , FILE *stream) {
  fprintf(stream,"DATES\n");
  {
    int i;
    for (i=0; i < time_t_vector_size( kw->date_list ); i++) {
      const time_t date = time_t_vector_iget( kw->date_list , i);
      time_t_to_dates_line_fprintf(date , stream);
    }
    fprintf(stream , "/\n\n");
  }
}



void sched_kw_dates_free(sched_kw_dates_type * kw) {
  time_t_vector_free(kw->date_list);
  free(kw);
}



void sched_kw_dates_fwrite(const sched_kw_dates_type *kw , FILE * stream) {
//  {
//    int date_lines = list_get_size(kw->date_list);
//    util_fwrite(&date_lines , sizeof date_lines , 1, stream , __func__);
//  }
//  {
//    list_node_type *date_node = list_get_head(kw->date_list);
//    while (date_node != NULL) {
//      const time_t * date = list_node_value_ptr(date_node);
//      util_fwrite(date, sizeof *date, 1, stream, __func__);
//      date_node = list_node_get_next(date_node);
//    }
//  }
//
}



sched_kw_dates_type * sched_kw_dates_fread_alloc(FILE * stream) {
//  int lines , line_nr;
//  sched_kw_dates_type *kw = sched_kw_dates_alloc() ;
//  util_fread(&lines, sizeof lines, 1, stream, __func__);
//  line_nr = 0;
//  while (line_nr < lines) {
//    time_t * date = util_malloc(sizeof * date, __func__);
//    util_fread(date, sizeof *date, 1, stream, __func__);
//    list_append_list_owned_ref(kw->date_list, date, free);
//    line_nr++;
//  } 
//  return kw;
  return NULL;
}



int sched_kw_dates_get_size(const sched_kw_dates_type * kw)
{
  return time_t_vector_size(kw->date_list);
}



sched_kw_dates_type * sched_kw_dates_alloc_from_time_t(time_t date)
{
  sched_kw_dates_type * kw = sched_kw_dates_alloc();
  time_t_vector_append(kw->date_list , date );
  return kw;
}



time_t sched_kw_dates_iget_time_t(const sched_kw_dates_type * kw, int i)
{
  return time_t_vector_iget( kw->date_list , i);
}



time_t sched_kw_dates_get_time_t(const sched_kw_dates_type * kw)
{
  if(sched_kw_dates_get_size(kw) > 1)
    util_abort("%s: Internal error, must use scehd_kw_dates_iget_time_t - aborting size:%d .\n", __func__ , sched_kw_dates_get_size(kw));
  
  return sched_kw_dates_iget_time_t(kw, 0);
}



/***********************************************************************/



KW_IMPL(dates)
