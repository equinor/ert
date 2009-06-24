#include <stdlib.h>
#include <stdbool.h>
#include <stringlist.h>
#include <util.h>
#include <sched_kw_wconprod.h>
#include <sched_kw_untyped.h>
#include <sched_util.h>




/**
   This file implements a very basic support for the WCONPROD
   keyword. It internalizes a list of well names, apart from that all
   information is just tucked into a untyped keyword.

   This means that all the functionality which is sppurted by the
   sched_kw_wconprod implementation is asking for well names.
*/
   




struct sched_kw_wconprod_struct {
  sched_kw_untyped_type * untyped_kw;
  stringlist_type       * wells;
};



static sched_kw_wconprod_type * sched_kw_wconprod_alloc(bool alloc_untyped)
{
  sched_kw_wconprod_type * kw = util_malloc(sizeof * kw, __func__);
  kw->wells      = stringlist_alloc_new();
  if (alloc_untyped)
    kw->untyped_kw = sched_kw_untyped_alloc("WCONPROD");  /* Hardcoded ... */
  else
    kw->untyped_kw = NULL;
  return kw;
}



void sched_kw_wconprod_free(sched_kw_wconprod_type * kw)
{
  stringlist_free(kw->wells);
  sched_kw_untyped_free(kw->untyped_kw);
}



static void sched_kw_wconprod_add_line(sched_kw_wconprod_type * kw , const char * line , FILE * stream) {
  int tokens;
  char ** token_list;
  bool    slash_term;

  sched_util_parse_line(line , &tokens , &token_list , 1 , &slash_term);
  if (!slash_term) 
    util_abort("%s: line[%d]: \"%s\" not properly terminated with \'/\' \n",__func__ , util_get_current_linenr(stream) , line);

  if (token_list[0] == NULL)
    util_abort("%s: line[%d]: failed to get well name \n",__func__ , util_get_current_linenr(stream));

  stringlist_append_copy(kw->wells , token_list[0]);
  sched_kw_untyped_add_line(kw->untyped_kw , line , NULL);
  util_free_stringlist( token_list , tokens );
}


sched_kw_wconprod_type * sched_kw_wconprod_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_wconprod_type * kw = sched_kw_wconprod_alloc(true);

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before WCONPROD was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_wconprod_add_line(kw, line , stream);
      free(line);
    }
  }
  return kw;
}


void sched_kw_wconprod_fwrite(const sched_kw_wconprod_type *kw , FILE *stream) {
  stringlist_fwrite( kw->wells , stream );
  sched_kw_untyped_fwrite( kw->untyped_kw , stream);
}


sched_kw_wconprod_type *  sched_kw_wconprod_fread_alloc(FILE *stream) {
  sched_kw_wconprod_type * kw = sched_kw_wconprod_alloc(false);
  
  stringlist_fread( kw->wells , stream );
  kw->untyped_kw = sched_kw_untyped_fread_alloc(stream);
  return kw;
  
}

void sched_kw_wconprod_fprintf(const sched_kw_wconprod_type * kw , FILE * stream) {
  sched_kw_untyped_fprintf( kw->untyped_kw , stream );
}


char ** sched_kw_wconprod_alloc_wells_copy( const sched_kw_wconprod_type * kw , int * num_wells) {
  *num_wells = stringlist_get_size( kw->wells );
  return stringlist_alloc_char_copy( kw->wells );
}


/*****************************************************************/

KW_IMPL(wconprod)
