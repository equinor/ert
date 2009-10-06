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



static sched_kw_dates_type * sched_kw_dates_alloc_empty()
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


static time_t parse_time_t(const char * day_string , const char * month_string , const char * year_string) {
  int mday , month , year;
  time_t time = -1;

  month = util_get_month_nr(month_string);
  if (month < 0)
    util_abort("%s: failed to interpret:%s a month name \n",__func__ , month_string );

  if (util_sscanf_int(day_string , &mday) && util_sscanf_int(year_string , &year))
    time = util_make_date(mday , month , year);
  else 
    util_abort("%s: fatal error when extracting date from:%s %s %s \n", __func__, day_string , month_string , year_string);

  return time;
}





static void time_t_to_dates_line_fprintf(time_t date, FILE * stream)
{
  int day, month, year;
  util_set_date_values(date, &day, &month, &year);
  fprintf(stream , "  %d \'%s\' %4d  /  \n" , day, get_month_string_from_int(month), year );
}





/*****************************************************************/


sched_kw_dates_type * sched_kw_dates_alloc(const stringlist_type * tokens , int * token_index ) {
  sched_kw_dates_type * kw = sched_kw_dates_alloc_empty();
  int eokw                    = false;
  do {
    stringlist_type * line_tokens = sched_util_alloc_line_tokens( tokens , false, 0 , token_index );
    if (line_tokens == NULL)
      eokw = true;
    else {
      if (stringlist_get_size( line_tokens ) == 3) {
        const char * day_string   = stringlist_iget( line_tokens , 0 );
        const char * month_string = stringlist_iget( line_tokens , 1 );
        const char * year_string  = stringlist_iget( line_tokens , 2 );

        time_t date = parse_time_t( day_string , month_string , year_string );
        time_t_vector_append( kw->date_list , date );
      } else {
        stringlist_fprintf( line_tokens , "  " , stdout );
        util_abort("%s: malformed DATES keyword\n",__func__);
      }
      stringlist_free( line_tokens );
    } 
    
  } while (!eokw);
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



int sched_kw_dates_get_size(const sched_kw_dates_type * kw)
{
  return time_t_vector_size(kw->date_list);
}



sched_kw_dates_type * sched_kw_dates_alloc_from_time_t(time_t date)
{
  sched_kw_dates_type * kw = sched_kw_dates_alloc_empty();
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


sched_kw_dates_type * sched_kw_dates_copyc(const sched_kw_dates_type * kw) {
  util_abort("%s: not implemented ... \n",__func__);
  return NULL;
}




/***********************************************************************/



KW_IMPL(dates)
     
