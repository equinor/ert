#include <stdlib.h>
#include <stdio.h>
#include <sched_kw_untyped.h>
#include <list.h>
#include <string.h>
#include <list_node.h>
#include <util.h>
#include <stdbool.h>


struct sched_kw_untyped_struct {
  bool       one_line;
  char      *kw_name;
  list_type *line_list;
};


/*****************************************************************/


void sched_kw_untyped_fprintf(const sched_kw_untyped_type *kw , FILE *stream) {
  fprintf(stream , "%s \n" , kw->kw_name);
  {
    list_node_type *line_node = list_get_head(kw->line_list);
    while (line_node != NULL) {
      const char * line = list_node_get_string(line_node);
      fprintf(stream , "   %s\n" , line);
      line_node = list_node_get_next(line_node);
    }

    if (kw->one_line)
      fprintf(stream,"\n");
    else
      fprintf(stream , "/\n\n");
    
  }
}


sched_kw_untyped_type * sched_kw_untyped_alloc(const char * kw_name , bool one_line) {
  sched_kw_untyped_type * kw = malloc(sizeof *kw);
  kw->line_list = list_alloc();
  kw->one_line  = one_line;
  kw->kw_name   = util_alloc_string_copy(kw_name);
  return kw;
}


void sched_kw_untyped_add_line(sched_kw_untyped_type * kw , const char *line) {
  list_append_string_copy(kw->line_list , line);
}



void sched_kw_untyped_free(sched_kw_untyped_type * kw) {
  list_free(kw->line_list);
  free(kw->kw_name);
  free(kw);
}



void sched_kw_untyped_fwrite(const sched_kw_untyped_type *kw , FILE *stream) {
  util_fwrite_string(kw->kw_name , stream);
  util_fwrite(&kw->one_line , sizeof kw->one_line , 1 , stream , __func__);
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
  bool one_line;
  char *kw_name = util_fread_alloc_string(stream);
  util_fread(&one_line , sizeof one_line , 1 , stream , __func__);
  {
    sched_kw_untyped_type *kw = sched_kw_untyped_alloc(kw_name , one_line);
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
  













