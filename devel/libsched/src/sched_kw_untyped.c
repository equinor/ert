#include <string.h>
#include <stdlib.h>
#include <list.h>
#include <list_node.h>
#include <ctype.h>
#include <util.h>
#include <sched_kw_untyped.h>
#include <sched_util.h>
#include <subst.h>




struct sched_kw_untyped_struct {
  char      *kw_name;  /* The name of the current keyword. */
  char      *buffer;   /* The content of the keyword is just appended in on char * pointer. */
};



/*****************************************************************/



static int get_fixed_record_length(const char * kw_name)
{
    
   if( strcmp(kw_name, "INCLUDE" ) == 0) { return  1;}
   if( strcmp(kw_name, "RPTSCHED") == 0) { return  1;}
   if( strcmp(kw_name, "DRSDT"   ) == 0) { return  1;}
   if( strcmp(kw_name, "SKIPREST") == 0) { return  0;}
   if( strcmp(kw_name, "RPTRST"  ) == 0) { return  1;}
   if( strcmp(kw_name, "TUNING"  ) == 0) { return  3;}
   if( strcmp(kw_name, "WHISTCTL") == 0) { return  1;}
   if( strcmp(kw_name, "TIME"    ) == 0) { return  1;}
   if( strcmp(kw_name, "VAPPARS" ) == 0) { return  1;}
   if( strcmp(kw_name, "NETBALAN") == 0) { return  1;}
   if( strcmp(kw_name, "WPAVE"   ) == 0) { return  1;}
    
   return -1;
}


sched_kw_untyped_type * sched_kw_untyped_alloc(const char * kw_name) {
  sched_kw_untyped_type * kw = util_malloc(sizeof *kw , __func__);
  kw->kw_name   = util_alloc_string_copy(kw_name);
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
    fseek(stream , 1 , SEEK_CUR); /* Skip the trailing '/' */
    if(line != NULL) {
      /** Add the terminating '/' explicitly ... */
      line = util_strcat_realloc(line , "/\n");
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
    int rec_len = get_fixed_record_length(kw->kw_name);
    if (kw->buffer != NULL)
      fprintf(stream , "%s" , kw->buffer);
    
    if(rec_len < 0)
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
    sched_kw_untyped_type *kw = sched_kw_untyped_alloc(kw_name);
    kw->buffer = util_fread_alloc_string(stream);
    return kw;
  }
}
  

/*****************************************************************/

KW_IMPL(untyped)









