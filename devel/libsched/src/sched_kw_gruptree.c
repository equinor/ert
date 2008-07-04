#include <hash.h>
#include <util.h>
#include <sched_kw_gruptree.h>
#include <sched_util.h>

struct sched_kw_gruptree_struct
{
  hash_type * gruptree_hash;
};



/***********************************************************************/



sched_kw_gruptree_type * sched_kw_gruptree_alloc()
{
  sched_kw_gruptree_type * kw = util_malloc(sizeof * kw, __func__);
  kw->gruptree_hash = hash_alloc();
  
  return kw;
};



void sched_kw_gruptree_free(sched_kw_gruptree_type * kw)
{
  hash_free(kw->gruptree_hash);
  free(kw);
};


void sched_kw_gruptree_fprintf(const sched_kw_gruptree_type * kw, FILE * stream)
{

  fprintf(stream, "GRUPTREE\n");
  
  {
    const char * child_name;
    const char * parent_name;
    child_name = hash_iter_get_first_key(kw->gruptree_hash);
    while(child_name != NULL)
    {
      parent_name = hash_get_string(kw->gruptree_hash,child_name);
      fprintf(stream,"  '%s'  '%s' /\n",child_name,parent_name);
      child_name = hash_iter_get_next_key(kw->gruptree_hash);

    }
    hash_iter_complete(kw->gruptree_hash);
  }

  fprintf(stream,"/\n\n");
};



void sched_kw_gruptree_add_line(sched_kw_gruptree_type * kw, const char * line)
{
  int tokens;
  char **token_list;

  sched_util_parse_line(line, &tokens, &token_list, 2, NULL);

  if(tokens > 2)
    util_abort("%s: Error when parsing record in GRUPTREE. Record must have one or two strings. Found %i - aborting.\n",__func__,tokens);
  
  if(token_list[1] == NULL)
    token_list[1] = "FIELD";
  
  hash_insert_string(kw->gruptree_hash,token_list[0],token_list[1]);
  util_free_stringlist( token_list , tokens );  /* Joakim la til denne */
};



void sched_kw_gruptree_fwrite(const sched_kw_gruptree_type * kw, FILE * stream)
{
  int gruptree_lines = hash_get_size(kw->gruptree_hash);
  util_fwrite(&gruptree_lines, sizeof gruptree_lines, 1, stream, __func__);

  {
    const char * child_name;
    const char * parent_name;
    child_name = hash_iter_get_first_key(kw->gruptree_hash);
    while(child_name != NULL)
    {
      parent_name = hash_get_string(kw->gruptree_hash,child_name);

      util_fwrite_string(child_name , stream);
      util_fwrite_string(parent_name, stream);

      child_name = hash_iter_get_next_key(kw->gruptree_hash);

    }
    hash_iter_complete(kw->gruptree_hash);
  }
};



sched_kw_gruptree_type * sched_kw_gruptree_fread_alloc(FILE * stream)
{
  int i, gruptree_lines;
  char * child_name;
  char * parent_name;

  sched_kw_gruptree_type * kw = sched_kw_gruptree_alloc();

  util_fread(&gruptree_lines, sizeof gruptree_lines, 1, stream, __func__);

  for(i=0; i<gruptree_lines; i++)
  {
    child_name  = util_fread_alloc_string(stream);
    parent_name = util_fread_alloc_string(stream);
    hash_insert_string(kw->gruptree_hash,child_name,parent_name);
    free(child_name);
    free(parent_name);
  }

  return kw;
};
