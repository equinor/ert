#include <string.h>
#include <stdlib.h>
#include <list.h>
#include <list_node.h>
#include <ctype.h>
#include <util.h>
#include <sched_kw_untyped.h>
#include <sched_util.h>
#include <subst.h>
#include <stringlist.h>



struct sched_kw_untyped_struct {
  int        rec_len;   
  char      *kw_name;  /* The name of the current keyword. */
  char      *buffer;   /* The content of the keyword is just appended in one char * pointer. */
};



/*****************************************************************/

static int get_fixed_record_length(const char * kw_name)
{
  
   if( strcmp(kw_name , "RPTSCHED") == 0) { return  1;}
   if( strcmp(kw_name , "DRSDT"   ) == 0) { return  1;}
   if( strcmp(kw_name , "SKIPREST") == 0) { return  0;}
   if( strcmp(kw_name , "RPTRST"  ) == 0) { return  1;}
   if( strcmp(kw_name , "TUNING"  ) == 0) { return  3;}
   if( strcmp(kw_name , "WHISTCTL") == 0) { return  1;}
   if( strcmp(kw_name , "TIME"    ) == 0) { return  1;}
   if( strcmp(kw_name , "VAPPARS" ) == 0) { return  1;}
   if( strcmp(kw_name , "NETBALAN") == 0) { return  1;}
   if( strcmp(kw_name , "WPAVE"   ) == 0) { return  1;}
   if( strcmp(kw_name , "VFPTABL" ) == 0) { return  1;}
   if( strcmp(kw_name , "GUIDERAT") == 0) { return  1;} 
   
   return -1;  /* Can not use 0 - because some KW actually have 0 as a valid fixed value. */
}



sched_kw_untyped_type * sched_kw_untyped_alloc(const char * kw_name) {
  sched_kw_untyped_type * kw = util_malloc(sizeof *kw , __func__);
  kw->kw_name   = util_alloc_string_copy(kw_name);
  kw->rec_len   = get_fixed_record_length( kw_name );
  kw->buffer    = NULL; 
  return kw;
}



/** This is exported for the keywords  which are just a minimum extension of untyped. */
void sched_kw_untyped_add_line(sched_kw_untyped_type * kw , const char *line, bool pad) {
  if (pad) {
    char * padded_line = util_alloc_sprintf("   %s\n" , line);
    kw->buffer = util_strcat_realloc(kw->buffer , padded_line);
    free(padded_line);
  } else
    kw->buffer = util_strcat_realloc(kw->buffer , line);
}



static sched_kw_untyped_type * sched_kw_untyped_fscanf_alloc_fixlen(FILE * stream, bool *at_eof, const char * kw_name, int rec_len)
{
  int cur_rec                = 0;
  sched_kw_untyped_type * kw = sched_kw_untyped_alloc(kw_name);
  
  while(cur_rec < rec_len) {
    char * line = sched_util_alloc_slash_terminated_line(stream);
    if(line != NULL) {
      sched_kw_untyped_add_line(kw, line , false );
      free(line);
    } else 
      util_exit("Something fishy - sched_util_alloc_slash_terminated() has returned NULL \n");
    
    cur_rec++;
  }
  return kw;
}



static sched_kw_untyped_type * sched_kw_untyped_fscanf_alloc_varlen(FILE * stream, bool *at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_untyped_type * kw = sched_kw_untyped_alloc(kw_name);

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      free(line);
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before %s was finished - aborting.\n", __func__, kw_name);
    }
    else
    {
      sched_kw_untyped_add_line(kw, line , true );
      free(line);
    }
  }

  return kw;
}


/*****************************************************************/


void sched_kw_untyped_add_tokens( sched_kw_untyped_type * kw , const stringlist_type * line_tokens) {
  char * line_buffer = stringlist_alloc_joined_string( line_tokens , "  ");
  sched_kw_untyped_add_line(kw, line_buffer , true );
  free( line_buffer );
}





sched_kw_untyped_type * sched_kw_untyped_token_alloc(const stringlist_type * tokens , int * token_index ) {
  const char * kw_name = NULL;

  /* First part - get hold of the kw name */
  {
    int kw_index = (*token_index) - 1;
    do {
      kw_name = stringlist_iget( tokens , kw_index);
      if (util_string_isspace( kw_name ))
        kw_name = NULL;  /* Try again */
      kw_index--;
    } while (kw_name == NULL && (kw_index >= 0));

    if (kw_name == NULL)
      util_abort("%s: internal error - failed to identify untyped kw name \n",__func__);
  }
  
  
  {
    bool eokw                  = false;
    sched_kw_untyped_type * kw = sched_kw_untyped_alloc( kw_name );
    int line_nr                = 0;
    do {
      stringlist_type * line_tokens = sched_util_alloc_line_tokens( tokens , true , 0 , token_index );
      if (line_tokens == NULL)
        eokw = true;
      else {
        sched_kw_untyped_add_tokens( kw , line_tokens );
        stringlist_free( line_tokens );
      }
      
      line_nr++;
      if (line_nr == kw->rec_len)
        eokw = true;
      
    } while (!eokw);
    return kw;
  }
}



sched_kw_untyped_type * sched_kw_untyped_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  int rec_len = get_fixed_record_length(kw_name);
  
  if(rec_len < 0 )
    return sched_kw_untyped_fscanf_alloc_varlen(stream, at_eof, kw_name);
  else
    return sched_kw_untyped_fscanf_alloc_fixlen(stream, at_eof, kw_name, rec_len);
}



void sched_kw_untyped_fprintf(const sched_kw_untyped_type *kw , FILE *stream) {
  fprintf(stream , "%s \n" , kw->kw_name);
  {
    if (kw->buffer != NULL)
      fprintf(stream , "%s" , kw->buffer);
    
    if(kw->rec_len < 0)
      fprintf(stream , "/\n\n");
    else
      fprintf(stream, "\n\n");
  }
}



void sched_kw_untyped_free(sched_kw_untyped_type * kw) {
  util_safe_free(kw->buffer);
  free(kw->kw_name);
  free(kw);
}



void sched_kw_untyped_fwrite(const sched_kw_untyped_type *kw , FILE *stream) {
  util_fwrite_string(kw->kw_name , stream);
  util_fwrite_string(kw->buffer , stream);
}



sched_kw_untyped_type * sched_kw_untyped_fread_alloc(FILE *stream) {
  char *kw_name = util_fread_alloc_string(stream);
  {
    sched_kw_untyped_type * kw = sched_kw_untyped_alloc(kw_name);
    kw->buffer = util_fread_alloc_string(stream);
    return kw;
  }
}
  

/*****************************************************************/

KW_IMPL(untyped)









