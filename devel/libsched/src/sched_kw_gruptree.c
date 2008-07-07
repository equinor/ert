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
    const int   num_keys = hash_get_size(kw->gruptree_hash);
    char ** child_list   = hash_alloc_keylist(kw->gruptree_hash);
    int i;

    for (i = 0; i < num_keys; i++) {
      const char * parent_name = hash_get_string(kw->gruptree_hash , child_list[i]);
      fprintf(stream,"  '%s'  '%s' /\n",child_list[i] , parent_name);
    }
    util_free_stringlist( child_list , num_keys );
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
    hash_insert_string(kw->gruptree_hash, token_list[0], "FIELD");
  else
    hash_insert_string(kw->gruptree_hash, token_list[0], token_list[1]);

  util_free_stringlist( token_list , tokens );
};



void sched_kw_gruptree_fwrite(const sched_kw_gruptree_type * kw, FILE * stream)
{
  int gruptree_lines = hash_get_size(kw->gruptree_hash);
  util_fwrite(&gruptree_lines, sizeof gruptree_lines, 1, stream, __func__);
  {
    const int   num_keys = hash_get_size(kw->gruptree_hash);
    char ** child_list   = hash_alloc_keylist(kw->gruptree_hash);
    int i;

    for (i = 0; i < num_keys; i++) {
      const char * parent_name = hash_get_string(kw->gruptree_hash , child_list[i]);

      util_fwrite_string(child_list[i] , stream);
      util_fwrite_string(parent_name   , stream);
    }
    util_free_stringlist( child_list , num_keys );
  }
}



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
