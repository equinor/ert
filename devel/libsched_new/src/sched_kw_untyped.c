#include <string.h>
#include <stdlib.h>
#include <list.h>
#include <list_node.h>
#include <util.h>
#include <sched_kw_untyped.h>
#include <sched_util.h>


struct sched_kw_untyped_struct {
  char      *kw_name;
  list_type *line_list;
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
    
   return -1;
}



static sched_kw_untyped_type * sched_kw_untyped_alloc(const char * kw_name) {
  sched_kw_untyped_type * kw = malloc(sizeof *kw);
  kw->line_list = list_alloc();
  kw->kw_name   = util_alloc_string_copy(kw_name);
  return kw;
}



static void sched_kw_untyped_add_line(sched_kw_untyped_type * kw , const char *line) {
  list_append_string_copy(kw->line_list , line);
}



static sched_kw_untyped_type * sched_kw_untyped_fscanf_alloc_fixlen(FILE * stream, bool *at_eof, const char * kw_name, int rec_len)
{
  int cur_rec = 0;
  bool at_eokw = false;
  char * line;
  sched_kw_untyped_type * kw = sched_kw_untyped_alloc(kw_name);

  while(!*at_eof && cur_rec < rec_len)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(*at_eof)
    {
      util_abort("%s: Reached EOF before %s was finished - aborting.\n", __func__, kw_name);
    }
    else
    {
      sched_kw_untyped_add_line(kw, line);
      cur_rec++;
      free(line);
    }
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
      sched_kw_untyped_add_line(kw, line);
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
    list_node_type *line_node = list_get_head(kw->line_list);
    while (line_node != NULL) {
      const char * line = list_node_get_string(line_node);
      fprintf(stream , "   %s\n" , line);
      line_node = list_node_get_next(line_node);
    }
  }
  {
    int rec_len = get_fixed_record_length(kw->kw_name);
    if(rec_len < 0)
      fprintf(stream , "/\n\n");
    else
      fprintf(stream, "\n\n");
  }
}



void sched_kw_untyped_free(sched_kw_untyped_type * kw) {
  list_free(kw->line_list);
  free(kw->kw_name);
  free(kw);
}



void sched_kw_untyped_fwrite(const sched_kw_untyped_type *kw , FILE *stream) {
  util_fwrite_string(kw->kw_name , stream);
  {
    int lines = list_get_size(kw->line_list);
    util_fwrite(&lines , sizeof lines , 1, stream , __func__);
  }
  {
    list_node_type *line_node = list_get_head(kw->line_list);
    while (line_node != NULL) {
      util_fwrite_string(list_node_get_string(line_node) , stream);
      /*printf("Skriver: <%s> %d \n",list_node_get_string(line_node) , strlen(list_node_get_string(line_node)));*/
      line_node = list_node_get_next(line_node);
    }
  }
}



sched_kw_untyped_type * sched_kw_untyped_fread_alloc(FILE *stream) {
  char *kw_name = util_fread_alloc_string(stream);
  {
    sched_kw_untyped_type *kw = sched_kw_untyped_alloc(kw_name);
    int lines , i;
    util_fread(&lines       , sizeof lines       , 1 , stream , __func__);
    for (i=0; i < lines; i++) {
      char * line = util_fread_alloc_string(stream);
      list_append_string_copy(kw->line_list , line);
      free(line);
    } 
    free(kw_name);
    return kw;
  }
}
  

/*
  This function tries to return the i'th (0 based) entry on each line
  in the kw.

  EXAMPLE
  ------------------


  Entry:
  -------------------------
  WCONPROD
  -- 0     1       2    ... 
     WP1  OPEN   ORAT   ... /
     WP2  OPEN   ORAT   ... /
  /

  Then sched_kw_untyped_iget_entries_alloc( ... , 0) would return WP1 and WP2.
*/
char ** sched_kw_untyped_iget_entries_alloc(const sched_kw_untyped_type * kw, int i, int * kw_size)
{
  int size = list_get_size(kw->line_list);
  char ** kw_entries = util_malloc(size * sizeof * kw_entries, __func__);
  for(int j=0; j<size; j++)
  {
    list_node_type * node = list_iget_node(kw->line_list,j);
    const char * line = list_node_get_string(node);
    int tokens;
    char ** token_list;
    sched_util_parse_line(line, &tokens, &token_list, i+1, NULL);
    kw_entries[j] = util_alloc_string_copy(token_list[i]);
    util_free_stringlist(token_list, tokens);
  }
  *kw_size = size;
  return kw_entries;
}

/*****************************************************************/

KW_FSCANF_ALLOC_IMPL(untyped)
KW_FWRITE_IMPL(untyped)
KW_FREAD_ALLOC_IMPL(untyped)
KW_FREE_IMPL(untyped)
KW_FPRINTF_IMPL(untyped)









