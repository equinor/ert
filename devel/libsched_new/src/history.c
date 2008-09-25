#include <string.h>
#include <time.h>
#include <util.h>
#include <hash.h>
#include <list.h>
#include <gruptree.h>
#include <history.h>

typedef struct history_node_struct history_node_type;

struct history_node_struct{
  /* Remember to fix history_node_copyc etc. if you add stuff here. */

  hash_type * well_hash;        /* A hash indexed with the well names.
                                   Each element is another hash, indexed
                                   by observations, where each element
                                   is a double.
                                */
  gruptree_type * gruptree;
  time_t      node_start_time;
  time_t      node_end_time;
};



struct history_struct{
  list_type   * nodes;
};



/******************************************************************/
// Functions for manipulating well_hash_type.


static void well_hash_fwrite(hash_type * well_hash, FILE * stream)
{
  int num_wells = hash_get_size(well_hash);
  char ** well_list = hash_alloc_keylist(well_hash);

  util_fwrite(&num_wells, sizeof num_wells, 1, stream, __func__);
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    util_fwrite_string(well_list[well_nr], stream);
    hash_type * well_obs_hash = hash_get(well_hash, well_list[well_nr]);

    int num_obs = hash_get_size(well_obs_hash);
    char ** var_list = hash_alloc_keylist(well_obs_hash);

    util_fwrite(&num_obs, sizeof num_obs, 1, stream, __func__);
    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
    {
      double obs = hash_get_double(well_obs_hash, var_list[obs_nr]);
      util_fwrite_string(var_list[obs_nr], stream);
      util_fwrite(&obs, sizeof obs, 1, stream, __func__);
    }
    util_free_stringlist(var_list, num_obs);
  }
  util_free_stringlist(well_list, num_wells);
}



static hash_type * well_hash_fread_alloc(FILE * stream)
{
  hash_type * well_hash = hash_alloc();
  int num_wells;

  util_fread(&num_wells, sizeof num_wells, 1, stream, __func__);
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    hash_type * well_obs_hash = hash_alloc();
    char * well_name = util_fread_alloc_string(stream);

    int num_obs;
    util_fread(&num_obs, sizeof num_obs, 1, stream, __func__);
    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
    {
      double obs;
      char * obs_name = util_fread_alloc_string(stream);
      util_fread(&obs, sizeof obs, 1, stream, __func__);
      hash_insert_double(well_obs_hash, obs_name, obs);
      free(obs_name);
    }

    hash_insert_hash_owned_ref(well_hash, well_name, well_obs_hash, hash_free__);
    free(well_name);
  }

  return well_hash;
}



static void well_hash_fprintf(hash_type * well_hash)
{
  int num_wells = hash_get_size(well_hash);
  char ** well_list = hash_alloc_keylist(well_hash);

  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    printf("WELL %s\n", well_list[well_nr]);
    printf("------------------------------------\n");
    hash_type * well_obs = hash_get(well_hash, well_list[well_nr]);
    int num_obs = hash_get_size(well_obs);
    char ** obs_list = hash_alloc_keylist(well_obs);
    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
      printf("%s : %f\n", obs_list[obs_nr], hash_get_double(well_obs, obs_list[obs_nr]));

    printf("------------------------------------\n\n");
    util_free_stringlist(obs_list, num_obs);
  }
  util_free_stringlist(well_list, num_wells);
}



static hash_type * well_hash_copyc(hash_type * well_hash_org)
{
  hash_type * well_hash_new = hash_alloc();

  int num_wells = hash_get_size(well_hash_org);
  char ** well_list = hash_alloc_keylist(well_hash_org);

  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    hash_type * well_obs_org = hash_get(well_hash_org, well_list[well_nr]);
    hash_type * well_obs_new = hash_alloc();

    int num_obs = hash_get_size(well_obs_org);
    char ** obs_list = hash_alloc_keylist(well_obs_org);
    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
    {
      double obs = hash_get_double(well_obs_org, obs_list[obs_nr]);
      hash_insert_double(well_obs_new, obs_list[obs_nr], obs);
    }
    hash_insert_hash_owned_ref(well_hash_new, well_list[well_nr], well_obs_new, hash_free__);

    util_free_stringlist(obs_list, num_obs);
  }
  util_free_stringlist(well_list, num_wells);

  return well_hash_new;
}



static hash_type * well_hash_alloc_from_summary(const ecl_sum_type * summary, 
                                                const char ** well_list, int num_wells, int restart_nr,
                                                bool use_h_keywords)
{
  hash_type * well_hash = hash_alloc();

  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    hash_type * well_obs = hash_alloc();

    // Cleaner than macros.
    void insert_obs(const char * well_name, const char * obs_name)
    {
      if(ecl_sum_has_well_var(summary, well_name, obs_name));
      {
        double obs = ecl_sum_get_well_var(summary, restart_nr, well_name, obs_name);
        hash_insert_double(well_obs, obs_name, obs);
      }
    }
    void insert_obs_use_h(const char * well_name, const char * obs_name, const char * obs_ins_name)
    {
      if(ecl_sum_has_well_var(summary, well_name, obs_name));
      {
        double obs = ecl_sum_get_well_var(summary, restart_nr, well_name, obs_name);
        hash_insert_double(well_obs, obs_ins_name, obs);
      }
    }

    if(use_h_keywords)
    {
      insert_obs(well_list[well_nr], "WOPR");
      insert_obs(well_list[well_nr], "WWPR");
      insert_obs(well_list[well_nr], "WGPR");
      insert_obs(well_list[well_nr], "WBHP");
      insert_obs(well_list[well_nr], "WTHP");
      insert_obs(well_list[well_nr], "WWCT");
      insert_obs(well_list[well_nr], "WGOR");
    }
    else
    {
      insert_obs_use_h(well_list[well_nr], "WOPRH", "WOPR");
      insert_obs_use_h(well_list[well_nr], "WWPRH", "WWPR");
      insert_obs_use_h(well_list[well_nr], "WGPRH", "WGPR");
      insert_obs_use_h(well_list[well_nr], "WBHPH", "WBHP");
      insert_obs_use_h(well_list[well_nr], "WTHPH", "WTHP");
      insert_obs_use_h(well_list[well_nr], "WWCTH", "WWCT");
      insert_obs_use_h(well_list[well_nr], "WGORH", "WGOR");
    }

    hash_insert_hash_owned_ref(well_hash, well_list[well_nr], well_obs, hash_free__);
  }

  return well_hash;
}



static double well_hash_get_var(hash_type * well_hash, const char * well, const char * var, bool * default_used)
{
  if(!hash_has_key(well_hash, well))
  {
    *default_used = true;
    return 0.0;
  }

  hash_type * well_obs = hash_get(well_hash, well);
  if(!hash_has_key(well_obs, var))
  {
    *default_used = true;
    return 0.0;
  }
  else
  {
    *default_used = false;
    return  hash_get_double(well_obs, var);
  }

}




/******************************************************************/
// Functions for manipulating history_node_type.


static history_node_type * history_node_alloc_empty()
{
  history_node_type * node = util_malloc(sizeof * node, __func__);
  node->well_hash          = hash_alloc();
  node->gruptree           = gruptree_alloc();
  return node;
}



static void history_node_free(history_node_type * node)
{
  hash_free(node->well_hash);
  gruptree_free(node->gruptree);
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
  well_hash_fwrite(node->well_hash, stream);
  gruptree_fwrite(node->gruptree, stream);
}



static history_node_type * history_node_fread_alloc(FILE * stream)
{
  history_node_type * node = util_malloc(sizeof * node, __func__);

  util_fread(&node->node_start_time, sizeof node->node_start_time, 1, stream, __func__);
  util_fread(&node->node_end_time,   sizeof node->node_end_time,   1, stream, __func__);

  node->well_hash = well_hash_fread_alloc(stream);
  node->gruptree  = gruptree_fread_alloc(stream);

  return node;
}



static void history_node_register_wells(history_node_type * node, hash_type * well_hash)
{
  int num_wells = hash_get_size(well_hash);
  char ** well_list = hash_alloc_keylist(well_hash);

  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    if(hash_has_key(node->well_hash, well_list[well_nr]))
      hash_del(node->well_hash, well_list[well_nr]);

    hash_type * well_obs_ext = hash_get(well_hash, well_list[well_nr]);
    hash_type * well_obs_int = hash_alloc();

    int num_obs = hash_get_size(well_obs_ext);
    char ** obs_list = hash_alloc_keylist(well_obs_ext);
    
    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
    {
      double obs = hash_get_double(well_obs_ext, obs_list[obs_nr]);
      hash_insert_double(well_obs_int, obs_list[obs_nr], obs);
    }
    util_free_stringlist(obs_list, num_obs);
    hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_int, hash_free__);
  }

  util_free_stringlist(well_list, num_wells);
}



static void history_node_delete_wells(history_node_type * node, const char ** well_list, int num_wells)
{
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    if(hash_has_key(node->well_hash, well_list[well_nr]))
    {
      hash_del(node->well_hash, well_list[well_nr]);
    }
  }
}



static void history_node_update_gruptree_grups(history_node_type * node, char ** children, char ** parents, int num_pairs)
{
  for(int pair = 0; pair < num_pairs; pair++)
    gruptree_register_grup(node->gruptree, children[pair], parents[pair]);
}



static void history_node_update_gruptree_wells(history_node_type * node, char ** children, char ** parents, int num_pairs)
{
  for(int pair = 0; pair < num_pairs; pair++)
    gruptree_register_well(node->gruptree, children[pair], parents[pair]);
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
      hash_type * well_hash = sched_kw_alloc_well_obs_hash(sched_kw);
      history_node_register_wells(node, well_hash);
      hash_free(well_hash);
      break;
    }
    case(WCONPROD):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJ):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJE):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJH):
    {
      hash_type * well_hash = sched_kw_alloc_well_obs_hash(sched_kw);
      history_node_register_wells(node, well_hash);
      hash_free(well_hash);
      break;
    }
    case(GRUPTREE):
    {
      int num_pairs;
      char ** children = NULL;
      char ** parents = NULL;
      sched_kw_alloc_child_parent_list(sched_kw, &children, &parents, &num_pairs);
      history_node_update_gruptree_grups(node, children, parents, num_pairs);
      util_free_stringlist(children, num_pairs);
      util_free_stringlist(parents, num_pairs);
      break;
    }
    case(WELSPECS):
    {
      int num_pairs;
      char ** children = NULL;
      char ** parents = NULL;
      sched_kw_alloc_child_parent_list(sched_kw, &children, &parents, &num_pairs);
      history_node_update_gruptree_wells(node, children, parents, num_pairs);
      util_free_stringlist(children, num_pairs);
      util_free_stringlist(parents, num_pairs);
      break;
    }
    default:
      break;
  }
}



static history_node_type * history_node_copyc(const history_node_type * node_org)
{
  history_node_type * node_new = util_malloc(sizeof * node_new, __func__);

  node_new->node_start_time    = node_org->node_start_time;
  node_new->node_end_time      = node_org->node_end_time;

  node_new->well_hash = well_hash_copyc(node_org->well_hash);
  node_new->gruptree  = gruptree_copyc(node_org->gruptree);

  return node_new;
}



/******************************************************************/
// Static functions for manipulating history_type.


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



static history_node_type * history_iget_node_ref(const history_type * history, int i)
{
  history_node_type * node = list_iget_node_value_ptr(history->nodes, i);
  return node;
}



/******************************************************************/
// Exported functions for manipulating history_type. Acess functions further below.


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

  int num_restart_files = sched_file_get_num_restart_files(sched_file);

  history_node_type * node = NULL;
  for(int block_nr = 0; block_nr < num_restart_files; block_nr++)
  {
    if(node != NULL)
    {
      history_node_type * node_cpy = history_node_copyc(node);
      node = node_cpy;
    }
    else
      node = history_node_alloc_empty();

    node->node_start_time = sched_file_iget_block_start_time(sched_file, block_nr);
    node->node_end_time   = sched_file_iget_block_end_time(sched_file, block_nr);

    int num_kws = sched_file_iget_block_size(sched_file, block_nr);
    for(int kw_nr = 0; kw_nr < num_kws; kw_nr++)
    {
      sched_kw_type * sched_kw = sched_file_ijget_block_kw_ref(sched_file, block_nr, kw_nr);
      history_node_parse_data_from_sched_kw(node, sched_kw);
    }
    history_add_node(history, node);
  }
  return history;
}



void history_realloc_from_summary(history_type * history, const ecl_sum_type * summary, bool use_h_keywords)
{
  int first_restart, last_restart, num_restarts;
  time_t current_time = ecl_sum_get_start_time(summary);
  ecl_sum_get_report_size(summary, &first_restart, &last_restart);
  num_restarts = history_get_num_restarts(history);

  // We don't take the pain to support missing restarts.
  if(first_restart > 0)
    util_abort("%s: Summary object does not contain the first %d restarts. Aborting.\n", __func__, first_restart);

  // We demand that the summary has been generated from the schedule used to alloc history.
  if(last_restart != num_restarts)
    util_abort("%s: Schedule file had %i restarts and summary file had %i, non-compatible.", __func__, num_restarts, last_restart);

  // OK, we are good to go.
  int     num_wells = ecl_sum_get_Nwells(summary);
  const char ** well_list = ecl_sum_get_well_names_ref(summary);

  // Special case for the first restart.
  history_node_type * node = list_iget_node_value_ptr(history->nodes, 0);
  node->node_start_time = current_time;
  node->node_end_time   = current_time;
  hash_free(node->well_hash);
  node->well_hash = well_hash_alloc_from_summary(summary, well_list, num_wells, 0, use_h_keywords);

  for(int block_nr = 1; block_nr <= last_restart; block_nr++)
  {
    node = list_iget_node_value_ptr(history->nodes, 0);
    node->node_start_time = current_time;
    current_time = ecl_sum_get_sim_time(summary, block_nr);
    node->node_end_time = current_time;
    hash_free(node->well_hash);
    node->well_hash = well_hash_alloc_from_summary(summary, well_list, num_wells, block_nr, use_h_keywords);
  }
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



/******************************************************************/
// Exported functions for accessing history_type.



int history_get_num_restarts(const history_type * history)
{
  return list_get_size(history->nodes);
}



double history_get_well_var(const history_type * history, int restart_num, const char * well, const char * var, bool * default_used)
{
  history_node_type * node = history_iget_node_ref(history, restart_num);
  return well_hash_get_var(node->well_hash, well, var, default_used);
}


double history_get_group_var(const history_type * history, int restart_num, const char * group, const char * var, bool * default_used)
{
  // TODO
  // Need to rewrite this to support both summary/schedule alloc'ed histories.
  history_node_type * node = history_iget_node_ref(history, restart_num);

  if(!gruptree_has_grup(node->gruptree, group))
  {
    *default_used = true;
    return 0.0;
  }

  char * wvar = NULL;
  if(     strcmp(var, "GOPR") == 0)
    wvar = "WOPR";
  else if(strcmp(var, "GWPR") == 0)
    wvar = "WWPR";
  else if(strcmp(var, "GGPR") == 0)
    wvar = "WGPR";
  else if(strcmp(var, "GOIR") == 0)
    wvar = "WOIR";
  else if(strcmp(var, "GWIR") == 0)
    wvar = "WWIR";
  else if(strcmp(var, "GGIR") == 0)
    wvar = "WGIR";
  else
  {
    util_abort("%s: No support for calculating group keyword %s from well keywords.\n", __func__);
  }

  double obs = 0.0;
  int num_wells;
  char ** well_list = gruptree_alloc_grup_well_list(node->gruptree, group, &num_wells);
  *default_used = false;
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    bool def = false;
    double obs_inc = well_hash_get_var(node->well_hash, well_list[well_nr], wvar, &def);
    obs = obs + obs_inc;
    if(def)
      *default_used = true;
  }
  util_free_stringlist(well_list, num_wells);
  return obs;
}
