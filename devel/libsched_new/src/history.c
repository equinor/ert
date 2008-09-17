#include <time.h>
#include <util.h>
#include <hash.h>
#include <list.h>
#include <history.h>
#include <rate.h>

typedef struct history_node_struct history_node_type;

struct history_node_struct{
  /* Remember to fix history_node_copyc if you add stuff here. */

  hash_type * rate_hash;        /* A hash of rate_type's. */
  time_t      node_start_time;
  time_t      node_end_time;
};



struct history_struct{
  list_type * nodes;
};



static history_node_type * history_node_alloc_empty()
{
  history_node_type * node = util_malloc(sizeof * node, __func__);
  node->rate_hash          = hash_alloc();
  return node;
}



static void history_node_free(history_node_type * node)
{
  hash_free(node->rate_hash);
  free(node);
}



static void history_node_free__(void * node)
{
  history_node_free( (history_node_type *) node);
}



static void history_node_fwrite(const history_node_type * node, FILE * stream)
{
  util_abort("%s: Not implemented.\n", __func__);  
}



static history_node_type * history_node_fread_alloc(FILE * stream)
{
  util_abort("%s: Not implemented.\n", __func__);  
  return NULL;
}



static void history_node_parse_data_from_sched_kw(history_node_type * node, const sched_kw_type * sched_kw)
{
  printf("%s: Warning - not implemented.\n", __func__); 
}



static history_node_type * history_node_copyc(const history_node_type * history_node)
{
  history_node_type * history_node_new = history_node_alloc_empty();
  history_node_new->node_start_time    = history_node->node_start_time;
  history_node_new->node_end_time      = history_node->node_end_time;

  {
    int size = hash_get_size(history_node->rate_hash);
    char ** keylist = hash_alloc_keylist(history_node->rate_hash);

    for(int i=0; i<size; i++)
    {
      char * key = keylist[i];
      rate_type * rate_old = hash_get(history_node->rate_hash, key);
      rate_type * rate_new = rate_copyc(rate_old);
      hash_insert_hash_owned_ref(history_node_new->rate_hash, key, rate_new, rate_free__);
    }

    util_free_stringlist(keylist, size);
  }
  

  return NULL;
}



/******************************************************************/



static history_type * history_alloc_empty()
{
  history_type * history = util_malloc(sizeof * history, __func__);
  history->nodes         = list_alloc();
  return history;
}



static void history_add_node(history_type * history, history_node_type * node)
{
  list_append_list_owned_ref(history->nodes, node, history_node_free__);
}



/******************************************************************/



void history_free(history_type * history)
{
  list_free(history->nodes);
  free(history);
}



/*
  The function history_alloc_from_sched_file tries to create
  a consistent history_type from a sched_file_type. Now, history_type
  and sched_file_type differ in one fundamental way which complicates
  this process.

  -----------------------------------------------------------------------
  The history_type is a "state object", i.e. all relevant information
  for a given block is contained in the corresponding history_node_type
  for the block. The sched_file_type however, is a "sequential object",
  where all information up to and including the current block is relevant.
  -----------------------------------------------------------------------

  Thus, to create a history_type object from a sched_file_type object,
  we must accumulate the changes in the sched_file_type object.
*/
history_type * history_alloc_from_sched_file(const sched_file_type * sched_file)
{
  history_type * history = history_alloc_empty();
  int num_restart_files = sched_file_get_nr_restart_files(sched_file);

  history_node_type * history_node = NULL;
  for(int block_nr = 0; block_nr < num_restart_files; block_nr++)
  {
    if(history_node == NULL)
      history_node = history_node_alloc_empty();
    else
      history_node = history_node_copyc(history_node);

    history_node_type * history_node = history_node_alloc_empty();
    history_node->node_start_time = sched_file_iget_block_start_time(sched_file, block_nr);
    history_node->node_end_time   = sched_file_iget_block_end_time(sched_file, block_nr);

    int num_kws = sched_file_iget_block_size(sched_file, block_nr);
    for(int kw_nr = 0; kw_nr < num_kws; kw_nr++)
    {
      sched_kw_type * sched_kw = sched_file_ijget_block_kw_ref(sched_file, block_nr, kw_nr);
      history_node_parse_data_from_sched_kw(history_node, sched_kw);
    }

    history_add_node(history, history_node);
  }
  return history;
}
