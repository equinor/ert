#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <hash.h>
#include <time.h>
#include <util.h>
#include <sched_kw.h>
#include <sched_util.h>
#include <sched_kw_gruptree.h>
#include <sched_kw_tstep.h>
#include <sched_kw_dates.h>
#include <sched_kw_wconhist.h>
#include <sched_kw_welspecs.h>
#include <sched_kw_untyped.h>
#include <sched_macros.h>


/*
  The structure sched_kw_type is used for internalization
  of arbitrary keywords in an ECLIPSE schedule file/section.

  Two structs are defined in this file:

    1. The sched_kw_type, which can be accessed externaly
       through various interface functions.
    2. The data_handlers_type, which provides an abstraction
       for the data_handling of the various keywords. This
       is for internal use only.

  Keywords from the ECLIPSE schedule are divided into three
  different groups:

    1. Fully internalized keywords, e.g. GRUPTREE.
       Functions implementing the data_handlers and
       more for these keywords are found in separate
       files, e.g. sched_kw_gruptree.c.

    2. Keywords which are known to have a fixed number
       of records, but not having a full internal
       representation. The number of records for these
       keywords are specified in the function
       get_fixed_record_length  in the file
       sched_kw_untyped.c

    3. Keywords which are not implemented and have a
       variable record length. These are handled 
       automatically by sched_kw_untyped.c.

*/

typedef void * (data_fscanf_alloc_proto)( FILE *, bool *,  const char *);
typedef void   (data_free_proto)(         void *);
typedef void   (data_fprintf_proto)(const void *, FILE *);
typedef void   (data_fwrite_proto)( const void *, FILE *);
typedef void * (data_fread_proto)(        FILE * );


typedef struct data_handlers_struct data_handlers_type;

struct data_handlers_struct {
  data_fscanf_alloc_proto * fscanf_alloc;
  data_free_proto         * free;
  data_fprintf_proto      * fprintf;
  data_fwrite_proto       * fwrite;
  data_fread_proto        * fread_alloc;
};



struct sched_kw_struct {
  sched_type_enum      type;
  data_handlers_type   data_handlers;
  void               * data;
};



/*****************************************************************/



static sched_type_enum get_sched_type_from_string(char * kw_name)
{
 if( strcmp(kw_name, GRUPTREE_STRING ) == 0){ return GRUPTREE ;}
 if( strcmp(kw_name, TSTEP_STRING    ) == 0){ return TSTEP    ;}
 if( strcmp(kw_name, TIME_STRING     ) == 0){ return TIME     ;}
 if( strcmp(kw_name, DATES_STRING    ) == 0){ return DATES    ;}
 if( strcmp(kw_name, WCONHIST_STRING ) == 0){ return WCONHIST ;}
 if( strcmp(kw_name, WELSPECS_STRING ) == 0){ return WELSPECS ;}
 else                                       { return UNTYPED  ;}
}



/*
  This tries to check if kw_name is a valid keyword
  in an ECLIPSE schedule file. Feel free to refine.
*/
static void sched_kw_name_assert(const char * kw_name)
{
  int tokens;
  char ** token_list;

  if(kw_name == NULL)
  {
    util_abort("%s: Internal error - trying to dereference NULL pointer.\n",__func__);
  }
  
  sched_util_parse_line(kw_name, &tokens, &token_list, 1, NULL);

  if(tokens != 1)
  {
      util_abort("%s: %s is not a valid schedule kw - aborting.\n",
                 __func__, kw_name);
  }
  sched_util_free_token_list(tokens, token_list);
  
}



/*
  This function returns the data_handlers_type for
  a specific sched_type_enum. If you implement a new
  type, be sure to add it here.
*/
static data_handlers_type get_data_handlers(sched_type_enum type)
{
  data_handlers_type handlers;
  
  switch(type) {
    case(GRUPTREE):
      GET_DATA_HANDLERS(handlers, gruptree);
      break;
    case(TSTEP):
      GET_DATA_HANDLERS(handlers, tstep);
      break;
    case(TIME):
      GET_DATA_HANDLERS(handlers, untyped);
      break;
    case(DATES):
      GET_DATA_HANDLERS(handlers, dates);
      break;
    case(WCONHIST):
      GET_DATA_HANDLERS(handlers, wconhist);
      break;
    case(WELSPECS):
      GET_DATA_HANDLERS(handlers, welspecs);
      break;
    case(UNTYPED):
      GET_DATA_HANDLERS(handlers, untyped);
      break;
    default:
      util_abort("%s: Internal error - aborting.\n",__func__);
  }
  
  return handlers;
}



static sched_kw_type ** sched_kw_tstep_split_alloc(const sched_kw_type * sched_kw, int * num_steps)
{
      *num_steps = sched_kw_tstep_get_size(sched_kw->data);
      sched_kw_type ** sched_kw_tsteps = util_malloc(*num_steps * sizeof * sched_kw_tsteps, __func__);

      for(int i=0; i<*num_steps; i++)
      {
        sched_kw_tsteps[i] = util_malloc(sizeof * sched_kw_tsteps[i], __func__);
        sched_kw_tsteps[i]->type = TSTEP;
        GET_DATA_HANDLERS(sched_kw_tsteps[i]->data_handlers, tstep);
        double step = sched_kw_tstep_iget_step((const sched_kw_tstep_type *) sched_kw->data, i);
        sched_kw_tsteps[i]->data = (void *) sched_kw_tstep_alloc_from_double(step);
      }
      
      return sched_kw_tsteps;
}



static sched_kw_type ** sched_kw_dates_split_alloc(const sched_kw_type * sched_kw, int * num_steps) 
{
      *num_steps = sched_kw_dates_get_size(sched_kw->data);
      sched_kw_type ** sched_kw_dates = util_malloc(*num_steps * sizeof * sched_kw_dates, __func__);

      for(int i=0; i<*num_steps; i++)
      {
        sched_kw_dates[i] = util_malloc(sizeof * sched_kw_dates[i], __func__);
        sched_kw_dates[i]->type = DATES;
        GET_DATA_HANDLERS(sched_kw_dates[i]->data_handlers, dates);
        time_t date = sched_kw_dates_iget_time_t((const sched_kw_dates_type *) sched_kw->data, i);
        sched_kw_dates[i]->data = (void *) sched_kw_dates_alloc_from_time_t(date);
      }
      return sched_kw_dates;
}
/*****************************************************************/



sched_type_enum sched_kw_get_type(const sched_kw_type * sched_kw)
{
  return sched_kw->type;  
}



/*
  This function will try to allocate a sched_kw from the schedule
  file  pointed by stream. The bool pointed by *at_eos will be
  set to true and NULL returned under the following conditions: 
    1. True EOF is reached.
    2. The keyword END is reached, terminating a valid schedule
       section.


  The function will abort under the following circumstances:
    1. If a valid kw is started, but EOF is reached prematurely.
    2. On encountering syntax errors in the schedule file, expect
       for missing END as the last keyword.


  *************************************************************
  ** Note that scheduling data after keyword END is ignored! **
  *************************************************************


*/
sched_kw_type * sched_kw_fscanf_alloc(FILE * stream, bool * at_eos)
{
  char * kw_name = NULL;
  *at_eos = false;
  while(kw_name == NULL && !*at_eos)
  {
    kw_name = sched_util_alloc_line(stream, at_eos);
  }
  
  if(*at_eos)
  {
    return NULL;
  }

  sched_kw_name_assert(kw_name);

  if(strcmp(kw_name,"END") == 0)
  {
    free(kw_name);
    *at_eos = true;
    return NULL;
  }

  {
    sched_kw_type * sched_kw = util_malloc(sizeof * sched_kw, __func__);
    sched_kw->type           = get_sched_type_from_string(kw_name);
    sched_kw->data_handlers  = get_data_handlers(sched_kw->type);
    sched_kw->data           = sched_kw->data_handlers.fscanf_alloc(stream, at_eos, kw_name); 

    free(kw_name);
    return sched_kw;
  } 
}



void sched_kw_free(sched_kw_type * sched_kw)
{
  sched_kw->data_handlers.free(sched_kw->data);
  free(sched_kw);
}



void sched_kw_free__(void * sched_kw_void)
{
  sched_kw_type * sched_kw = (sched_kw_type *) sched_kw_void;
  sched_kw_free(sched_kw);
}



/*
  This will print the kw in ECLIPSE style formating.
*/
void sched_kw_fprintf(const sched_kw_type * sched_kw, FILE * stream)
{
  sched_kw->data_handlers.fprintf(sched_kw->data, stream);
}



/*
  This stores the kw to stream in a format specified
  by each kw.
*/
void sched_kw_fwrite(sched_kw_type * sched_kw, FILE * stream)
{
  util_fwrite(&sched_kw->type, sizeof sched_kw->type, 1, stream, __func__);
  sched_kw->data_handlers.fwrite(sched_kw->data, stream);
}



/*
  This reads the kw from stream. It is assumed that the
  kw has previously been written using sched_kw_fwrite.
*/
sched_kw_type * sched_kw_fread_alloc(FILE * stream, bool * at_eof)
{
  sched_kw_type * sched_kw = util_malloc(sizeof * sched_kw, __func__);
  util_fread(&sched_kw->type, sizeof sched_kw->type, 1, stream, __func__);

  sched_kw->data_handlers = get_data_handlers(sched_kw->type);
  sched_kw->data = sched_kw->data_handlers.fread_alloc(stream);

  return sched_kw;
}




/*
  This function takes a kw related to timing, such as DATES or TSTEP
  and converts it into a series of kw's with one timing event in each kw.

  Note that TIME (ECL300 only) is not supported yet.
*/
sched_kw_type ** sched_kw_restart_file_split_alloc(const sched_kw_type * sched_kw,  int * num_steps)
{
  switch(sched_kw_get_type(sched_kw))
  {
    case(TSTEP):
      return sched_kw_tstep_split_alloc(sched_kw, num_steps);

    case(DATES):
      return sched_kw_dates_split_alloc(sched_kw, num_steps);

     case(TIME):
       util_abort("%s: Sorry - no support for TIME kw yet. Please use TSTEP.\n", __func__);
       return NULL;
     default:
       util_abort("%s: Internal error - aborting.\n", __func__);
       return NULL;
  }
}



time_t sched_kw_get_new_time(const sched_kw_type * sched_kw, time_t curr_time)
{
  switch(sched_kw_get_type(sched_kw))
  {
    case(TSTEP):
      return sched_kw_tstep_get_new_time((const sched_kw_tstep_type *) sched_kw->data, curr_time);

    case(DATES):
      return sched_kw_dates_get_time_t((const sched_kw_dates_type *) sched_kw->data);

     case(TIME):
       util_abort("%s: Sorry - no support for TIME kw yet. Please use TSTEP.\n", __func__);
       return 0;
     default:
       util_abort("%s: Internal error - trying to get time from non-timing kw - aborting.\n", __func__);
       return 0;
  }
}
