#include <time.h>
#include <util.h>
#include <hash.h>
#include <list.h>
#include <history.h>
#include <rate.h>

typedef struct history_node_struct history_node_type;

struct history_node_struct{
  /* Remember to fix history_node_copyc etc. if you add stuff here. */

  hash_type * rate_hash;        /* A hash of rate_type's. */
  time_t      node_start_time;
  time_t      node_end_time;
};



struct history_struct{
  list_type * nodes;
};



/******************************************************************/



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
  util_fwrite(&node->node_start_time, sizeof node->node_start_time, 1, stream, __func__);
  util_fwrite(&node->node_end_time,   sizeof node->node_end_time,   1, stream, __func__);

  /* Write rate_hash. */
  {
    int size = hash_get_size(node->rate_hash);
    char ** keylist = hash_alloc_keylist(node->rate_hash);

    util_fwrite(&size, sizeof size, 1, stream, __func__);

    for(int i=0; i<size; i++)
    {
      rate_type * rate =  hash_get(node->rate_hash, keylist[i]);
      rate_fwrite(rate, stream);
    }
    util_free_stringlist(keylist, size);
  }
}



static history_node_type * history_node_fread_alloc(FILE * stream)
{
  history_node_type * node = history_node_alloc_empty();

  util_fread(&node->node_start_time, sizeof node->node_start_time, 1, stream, __func__);
  util_fread(&node->node_end_time,   sizeof node->node_end_time,   1, stream, __func__);

  {
    int size;
    util_fread(&size, sizeof size, 1, stream, __func__);

    for(int i=0; i<size; i++)
    {
      rate_type * rate = rate_fread_alloc(stream);
      hash_insert_hash_owned_ref(node->rate_hash, rate_get_well_ref(rate), rate, rate_free__);
    }
  }

  return node;
}



static void history_node_update_rate_hash(history_node_type * node, hash_type * rate_hash_diff)
{
  int size = hash_get_size(rate_hash_diff);
  char ** keylist = hash_alloc_keylist(rate_hash_diff);

  for(int i=0; i<size; i++)
  {
    rate_type * new_ref = hash_get(rate_hash_diff, keylist[i]);
    rate_type * new_cpy = rate_copyc(new_ref);
    hash_insert_hash_owned_ref(node->rate_hash, keylist[i], new_cpy, rate_free__);
  }

  util_free_stringlist(keylist, size);
}



/*
  The function history_node_parse_data_from_sched_kw updates the
  history_node_type pointer node with data from sched_kw. I.e., if sched_kw
  indicates that a producer has been turned into an injector, rate
  information about the producer shall be removed from node if it exists.
*/
static void history_node_parse_data_from_sched_kw(history_node_type * node, const sched_kw_type * sched_kw)
{
  switch(sched_kw_get_type(sched_kw))
  {
    case(WCONHIST):
    {
      hash_type * rate_hash_diff  = sched_kw_rate_hash_copyc(sched_kw);
      history_node_update_rate_hash(node, rate_hash_diff); 
      hash_free(rate_hash_diff);
      break;
    }
    default:
      break;
  }
}



static history_node_type * history_node_copyc(const history_node_type * history_node)
{
  history_node_type * history_node_new = history_node_alloc_empty();

  history_node_new->node_start_time    = history_node->node_start_time;
  history_node_new->node_end_time      = history_node->node_end_time;

  /* Copy rate_hash. */
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
  

  return history_node_new;
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
    if(history_node != NULL)
      history_node = history_node_copyc(history_node);
    else
      history_node = history_node_alloc_empty();

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



void history_fwrite(const history_type * history, FILE * stream)
{
  int size = list_get_size(history->nodes);  
  util_fwrite(&size, sizeof size, 1, stream, __func__);

  for(int i=0; i<size; i++)
  {
    history_node_type * node = list_iget_node_value_ptr(history->nodes, i);
    history_node_fwrite(node, stream);
  }
}



history_type * history_fread_alloc(FILE * stream)
{
  history_type * history = history_alloc_empty();

  int size;
  util_fread(&size, sizeof size, 1, stream, __func__);

  for(int i=0; i<size; i++)
  {
    history_node_type * node = history_node_fread_alloc(stream);
    list_append_list_owned_ref(history->nodes, node, history_node_free__);
  }

  return history;
}
