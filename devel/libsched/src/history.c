#include <stdio.h>
#include <string.h>
#include <util.h>
#include <hash.h>
#include <list.h>
#include <ecl_sum.h>
#include <gruptree.h>
#include <history.h>

typedef struct history_node_struct history_node_type;

struct history_node_struct{
  /* Remember to fix history_node_copyc etc. if you add stuff here. */

  hash_type     * well_hash;        /* 
                                       A hash indexed with the well names.  Each element is another hash, indexed by
                                       observations, where each element is a double value.
                                    */
  gruptree_type * gruptree;
  time_t      	  node_start_time;
  time_t      	  node_end_time;
  int         	  restart_nr;
};



struct history_struct{
  list_type       * nodes;
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



/**
   The list of variables WOPR +++ is compiled in - hmmm.
*/

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
      if(ecl_sum_has_well_var(summary, well_name, obs_name)) {
        double obs = ecl_sum_get_well_var(summary, restart_nr, well_name, obs_name);
        hash_insert_double(well_obs, obs_name, obs);
      } 
    }
    
    void insert_obs_use_h(const char * well_name, const char * obs_name, const char * obs_ins_name)
      {
      if(ecl_sum_has_well_var(summary, well_name, obs_name)) {
	double obs = ecl_sum_get_well_var(summary, restart_nr, well_name, obs_name);
	hash_insert_double(well_obs, obs_ins_name, obs);
      }
    }

    if(!use_h_keywords)
    {
      insert_obs(well_list[well_nr], "WOPR");
      insert_obs(well_list[well_nr], "WWPR");
      insert_obs(well_list[well_nr], "WGPR");
      insert_obs(well_list[well_nr], "WBHP");
      insert_obs(well_list[well_nr], "WTHP");
      insert_obs(well_list[well_nr], "WWCT");
      insert_obs(well_list[well_nr], "WGOR");
      insert_obs(well_list[well_nr], "WOPT");
      insert_obs(well_list[well_nr], "WGPT");
      insert_obs(well_list[well_nr], "WWPT");
      insert_obs(well_list[well_nr], "WWIR");
      insert_obs(well_list[well_nr], "WWIT");
      insert_obs(well_list[well_nr], "WGIR");
      insert_obs(well_list[well_nr], "WGIT");
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
      insert_obs_use_h(well_list[well_nr], "WOPTH", "WOPT");
      insert_obs_use_h(well_list[well_nr], "WGPTH", "WGPT");
      insert_obs_use_h(well_list[well_nr], "WWPTH", "WOPT");
      insert_obs_use_h(well_list[well_nr], "WWIRH", "WWIR");
      insert_obs_use_h(well_list[well_nr], "WWITH", "WWIT");
      insert_obs_use_h(well_list[well_nr], "WGIRH", "WGIR");
      insert_obs_use_h(well_list[well_nr], "WGITH", "WGIT");
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

  {
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

}




/******************************************************************/
// Functions for manipulating history_node_type.


static history_node_type * history_node_alloc_empty()
{
  history_node_type * node = util_malloc(sizeof * node, __func__);
  node->well_hash          = hash_alloc();
  node->gruptree           = gruptree_alloc();
  node->restart_nr         = 0;
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



/*
  This function will add the observations from well_hash to the well_hash in
  the history_node. If the wells are already in the history_node, their data
  is replaced with the data from well_hash.
*/
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



/*
  This function will add default observations for production wells. This is neccessary, since
  the well_hash has no concept of well status.

  If some of the wells in well_list are not present, they are created.
*/
static void history_node_add_producer_defaults(history_node_type * node, const char ** well_list, int num_wells)
{
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    if(hash_has_key(node->well_hash, well_list[well_nr]))
    {
      hash_type * well_obs_hash = hash_get(node->well_hash, well_list[well_nr]);
      hash_insert_double(well_obs_hash, "WWIR", 0.0);
      hash_insert_double(well_obs_hash, "WGIR", 0.0);
      hash_insert_double(well_obs_hash, "WOIR", 0.0);
    }
    else
    {
      hash_type * well_obs_hash = hash_alloc();
      hash_insert_double(well_obs_hash, "WWIR", 0.0);
      hash_insert_double(well_obs_hash, "WGIR", 0.0);
      hash_insert_double(well_obs_hash, "WOIR", 0.0);
      hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_hash, hash_free__);
    }
  }
}



/*
  This function will add default observations for injection wells. This is neccessary, since
  the well_hash has no concept of well status.

  If some of the wells in well_list are not present, they are created.
*/
static void history_node_add_injector_defaults(history_node_type * node, const char ** well_list, int num_wells)
{
  for(int well_nr = 0; well_nr < num_wells; well_nr++)
  {
    if(hash_has_key(node->well_hash, well_list[well_nr]))
    {
      hash_type * well_obs_hash = hash_get(node->well_hash, well_list[well_nr]);
      hash_insert_double(well_obs_hash, "WOPR", 0.0);
      hash_insert_double(well_obs_hash, "WWPR", 0.0);
      hash_insert_double(well_obs_hash, "WGPR", 0.0);
      hash_insert_double(well_obs_hash, "WWGPR", 0.0);
      hash_insert_double(well_obs_hash, "WWCT", 0.0);
      hash_insert_double(well_obs_hash, "WGOR", 0.0);
    }
    else
    {
      hash_type * well_obs_hash = hash_alloc();
      hash_insert_double(well_obs_hash, "WOPR", 0.0);
      hash_insert_double(well_obs_hash, "WWPR", 0.0);
      hash_insert_double(well_obs_hash, "WGPR", 0.0);
      hash_insert_double(well_obs_hash, "WWGPR", 0.0);
      hash_insert_double(well_obs_hash, "WWCT", 0.0);
      hash_insert_double(well_obs_hash, "WGOR", 0.0);
      hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_hash, hash_free__);
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
      int num_wells = hash_get_size(well_hash);
      char ** well_list = hash_alloc_keylist(well_hash);
      history_node_add_producer_defaults(node, (const char **) well_list, num_wells);
      util_free_stringlist(well_list, num_wells);
      hash_free(well_hash);
      break;
    }
    case(WCONPROD):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      history_node_add_producer_defaults(node, (const char **) well_list, num_wells);
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJ):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJE):
    {
      int num_wells;
      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
      history_node_delete_wells(node, (const char **) well_list, num_wells); 
      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
      util_free_stringlist(well_list, num_wells);
      break;
    }
    case(WCONINJH):
    {
      hash_type * well_hash = sched_kw_alloc_well_obs_hash(sched_kw);
      history_node_register_wells(node, well_hash);
      int num_wells = hash_get_size(well_hash);
      char ** well_list = hash_alloc_keylist(well_hash);
      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
      util_free_stringlist(well_list, num_wells);
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
  node_new->restart_nr         = node_org->restart_nr; 

  node_new->well_hash  = well_hash_copyc(node_org->well_hash);
  node_new->gruptree   = gruptree_copyc(node_org->gruptree);

  return node_new;
}


static void history_node_fprintf(const history_node_type * node, FILE * stream) {
  fprintf(stream , "%03d:  " , node->restart_nr); util_fprintf_date( node->node_start_time , stream); fprintf(stream , " => "); util_fprintf_date(node->node_end_time , stream); fprintf(stream , "\n");
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
  history_type * history = history_alloc_empty( );

  int num_restart_files = sched_file_get_num_restart_files(sched_file);

  history_node_type * node = NULL;
  for(int block_nr = 0; block_nr < num_restart_files; block_nr++)
  {
    if(node != NULL) {
      history_node_type * node_cpy = history_node_copyc(node);
      node = node_cpy;
    }
    else
      node = history_node_alloc_empty();

    node->node_start_time = sched_file_iget_block_start_time(sched_file, block_nr);
    node->node_end_time   = sched_file_iget_block_end_time(sched_file, block_nr);
    node->restart_nr      = block_nr;

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




/** 
    This function will take an existing history object, which has been allocated
    from a schedule file, and replace all the rates with the corresponding rates
    from the summary object.
    
    Observe the following:

     1. The time-framework, i.e. when the blocks starts and stops is unchanged,
        i.e. the values from the original SCHEDULE file are retained.

     2. IFF the summary file has time information (which is essential for
        queries on dates), it is checked that the dates in the summary object
        match up with the SCHEDULE file.

     3. All the wells from the summary object are used, along with a (compiled
        in) list of variables. [This should be more flexible ...]
    
*/

void history_realloc_from_summary(history_type * history, const ecl_sum_type * summary, bool use_h_keywords)
{
  bool          has_sim_time  = ecl_sum_has_sim_time( summary );
  int           num_restarts  = history_get_num_restarts(history);
  int           num_wells     = ecl_sum_get_Nwells(summary);
  const char ** well_list     = ecl_sum_get_well_names_ref(summary);

  for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++) {
    /* The list elements in history->nodes are updated IN-PLACE. */
    history_node_type * node = list_iget_node_value_ptr(history->nodes, restart_nr);
    hash_free(node->well_hash);  /* Removing the old information. */


    if (ecl_sum_has_report_nr(summary , restart_nr)) 
    {
      if (has_sim_time) 
      {
        time_t sum_time = ecl_sum_get_sim_time( summary , restart_nr );
        if (sum_time != node->node_end_time) 
          util_abort("%s: Timing inconsisentcy between schedule_file and refcase:%s at restart %i\n. Did you remember the DATE keyword in the SUMMARY section?", __func__, ecl_sum_get_simulation_case( summary), restart_nr );
      }

      node->well_hash = well_hash_alloc_from_summary(summary, well_list, num_wells, restart_nr, use_h_keywords);
    } 
    else 
    {
      fprintf(stderr,"Warning: refcase: \'%s\' does not have any data for report_step: %d \n", ecl_sum_get_simulation_case( summary ) , restart_nr);
      node->well_hash = hash_alloc();  
    }
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




/**
  Get the number of restart files the underlying schedule file would produce.
*/
int history_get_num_restarts(const history_type * history)
{
  return list_get_size(history->nodes);
}



/**
  Return true if the string pointed by name is a well at the restart_nr.
*/
bool history_str_is_well_name(const history_type * history, int restart_nr, const char * name)
{
  history_node_type * node = history_iget_node_ref(history, restart_nr);
  return hash_has_key(node->well_hash, name);
}


/**
  Return true if the string pointed by name is a group at the restart_nr.
*/
bool history_str_is_group_name(const history_type * history, int restart_nr, const char * name)
{
  history_node_type * node = history_iget_node_ref(history, restart_nr);
  return gruptree_has_grup(node->gruptree, name);
}


/**
  This function takes a key in the same format as the summary.x program, e.g. GOPR:GROUPA, 
  and tries to return the observed value for that key.
*/
double history_get_var_from_summary_key(const history_type * history, int restart_nr, const char * summary_key, bool * default_used)
{
  int argc;
  char ** argv;
  double val = 0.0;

  util_split_string(summary_key, ":", &argc, &argv);

  if(argc != 2)
    util_abort("%s: Key \"%s\" does not appear to be a valid summary key.\n", __func__, summary_key);

  if(history_str_is_group_name(history, restart_nr, argv[1]))
    val = history_get_group_var(history, restart_nr, argv[1], argv[0], default_used);
  else if(history_str_is_well_name(history, restart_nr, argv[1]))
    val = history_get_well_var(history, restart_nr, argv[1], argv[0], default_used);
  else
    *default_used = true;

  util_free_stringlist(argv, argc);
  return val;
}


/**
  Get the observed value for a well var.
*/
double history_get_well_var(const history_type * history, int restart_nr, const char * well, const char * var, bool * default_used)
{
  history_node_type * node = history_iget_node_ref(history, restart_nr);
  return well_hash_get_var(node->well_hash, well, var, default_used);
}



/**
  Get the observed value for a group var.
*/
double history_get_group_var(const history_type * history, int restart_nr, const char * group, const char * var, bool * default_used)
{
  history_node_type * node = history_iget_node_ref(history, restart_nr);

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
    util_abort("%s: No support for calculating group keyword %s for group %s from well keywords.\n", __func__, var, group);
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



/**
  This function alloc's two vectors of length num_restarts from a summary key.
  In the vector value, the observed values or default values for each restart
  is stored. If the default value is used, the corresponding element in the
  vector default_used will be true.
*/
void   history_alloc_time_series_from_summary_key
(
                    const history_type * history,
                    const char         * summary_key,
                    int                * __num_restarts,
                    double            ** __value,
                    bool              ** __default_used
)
{
  int num_restarts = history_get_num_restarts(history);

  double * value        = util_malloc(num_restarts * sizeof * value,        __func__);
  bool   * default_used = util_malloc(num_restarts * sizeof * default_used, __func__);

  int argc;
  char ** argv;
  util_split_string(summary_key, ":", &argc, &argv);
  if(argc != 2)
    util_abort("%s: Key \"%s\" does not appear to be a valid summary key.\n", __func__, summary_key);

  for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++)
  {
    if(history_str_is_group_name(history, restart_nr, argv[1]))
      value[restart_nr] = history_get_group_var(history, restart_nr, argv[1], argv[0], &default_used[restart_nr]);
    else if(history_str_is_well_name(history, restart_nr, argv[1]))
      value[restart_nr] = history_get_well_var(history, restart_nr, argv[1], argv[0], &default_used[restart_nr]);
    else
      default_used[restart_nr] = true;
  }

  util_free_stringlist(argv, argc);
  *__num_restarts = num_restarts;
  *__value        = value;
  *__default_used = default_used;
}



time_t history_iget_node_start_time(const history_type * history, int node_nr)
{
  history_node_type * history_node = history_iget_node_ref(history, node_nr);
  return history_node->node_start_time;
}



time_t history_iget_node_end_time(const history_type * history, int node_nr)
{
  history_node_type * history_node = history_iget_node_ref(history, node_nr);
  return history_node->node_end_time;
}



int history_get_restart_nr_from_time_t(const history_type * history, time_t time)
{
  int num_restart_files = history_get_num_restarts(history);
  int restart_nr        = 0;

  while (restart_nr < num_restart_files) {
    time_t node_end_time = history_iget_node_end_time(history, restart_nr);
    
    if (node_end_time == time)  
      break;                    /* Got it */
    else
      restart_nr++;
  }
  
  if (restart_nr == num_restart_files) {
    int sec,min,hour;
    int mday,year,month;
    
    util_set_datetime_values(time , &sec , &min , &hour , &mday , &month , &year);
    util_abort("%s: Time variable:%02d/%02d/%4d  %02d:%02d:%02d   does not cooincide with any restart file. Aborting.\n", __func__ , mday,month,year,hour,min,sec);
  }
  return restart_nr;
}



int history_get_restart_nr_from_days(const history_type * history, double days)
{
  if(days < 0.0)
    util_abort("%s: Cannot find a restart nr from a negative production period (days was %g).\n",
               __func__, days);

  time_t time = history_iget_node_start_time(history, 0);
  util_inplace_forward_days(&time, days);

  return history_get_restart_nr_from_time_t(history, time);
}


void history_fprintf(const history_type * history , FILE * stream) {
  int item;
  for (item = 0; item < list_get_size(history->nodes); item++) {
    const history_node_type * node = list_iget_node_value_ptr(history->nodes , item);
    history_node_fprintf(node , stream);
  }
  
}



char ** history_alloc_well_list(const history_type * history, int * num_wells)
{
  char      ** well_list;
  hash_type  * wells        = hash_alloc();
  int          num_restarts = history_get_num_restarts(history);

  for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++)
  {
    history_node_type * node  = history_iget_node_ref(history, restart_nr);
    int     num_current_wells = hash_get_size(node->well_hash);
    char ** current_wells     = hash_alloc_keylist(node->well_hash);
    
    for(int well_nr = 0; well_nr < num_current_wells; well_nr++)
    {
      if(!hash_has_key(wells, current_wells[well_nr]))
        hash_insert_int(wells, current_wells[well_nr], 0);
    }
    util_free_stringlist(current_wells, num_current_wells);
  }

  if(num_wells == NULL)
    util_abort("%s: Trying to dereference NULL pointer.\n", __func__);

  *num_wells = hash_get_size(wells);
  well_list  = hash_alloc_keylist(wells);
  hash_free(wells);
  return well_list;
}


