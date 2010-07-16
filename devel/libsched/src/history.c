#include <stdio.h>
#include <string.h>
#include <util.h>
#include <hash.h>
#include <ecl_sum.h>
#include <ecl_util.h>
#include <stringlist.h>
#include <bool_vector.h>
#include <gruptree.h>
#include <sched_history.h>
#include <history.h>


struct history_struct{
  const ecl_sum_type    * ecl_sum;        /* ecl_sum instance used when the data are taken from a summary instance. Observe that this is NOT owned by history instance.*/
  sched_history_type    * sched_history;
  history_source_type     source;
};


history_source_type history_get_source_type( const char * string_source ) {
  history_source_type source_type;

  if (strcmp( string_source , "REFCASE_SIMULATED") == 0)
    source_type = REFCASE_SIMULATED;
  else if (strcmp( string_source , "REFCASE_HISTORY") == 0)
    source_type = REFCASE_HISTORY;
  else if (strcmp( string_source , "SCHEDULE") == 0)
    source_type = SCHEDULE;
  else
    util_abort("%s: Sorry source:%s not recognized\n",__func__ , string_source);

  return source_type;
}


const char * history_get_source_string( history_source_type history_source ) {
  switch( history_source ) {
  case( REFCASE_SIMULATED ):
    return "REFCASE_SIMULATED";
    break;
  case(REFCASE_HISTORY ):
    return "REFCASE_HISTORY";
    break;
  case(SCHEDULE ):
    return "SCHEDULE";
    break;
  default: 
    util_abort("%s: internal fuck up \n",__func__);
    return NULL;
  }
}



// Del:/******************************************************************/
// Del:// Functions for manipulating well_hash_type.
// Del:
// Del:
// Del:
// Del:
// Del:
// Del:static void well_hash_fprintf(hash_type * well_hash)
// Del:{
// Del:  int num_wells = hash_get_size(well_hash);
// Del:  char ** well_list = hash_alloc_keylist(well_hash);
// Del:
// Del:  for(int well_nr = 0; well_nr < num_wells; well_nr++)
// Del:  {
// Del:    printf("WELL %s\n", well_list[well_nr]);
// Del:    printf("------------------------------------\n");
// Del:    hash_type * well_obs = hash_get(well_hash, well_list[well_nr]);
// Del:    int num_obs = hash_get_size(well_obs);
// Del:    char ** obs_list = hash_alloc_keylist(well_obs);
// Del:    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
// Del:      printf("%s : %f\n", obs_list[obs_nr], hash_get_double(well_obs, obs_list[obs_nr]));
// Del:
// Del:    printf("------------------------------------\n\n");
// Del:    util_free_stringlist(obs_list, num_obs);
// Del:  }
// Del:  util_free_stringlist(well_list, num_wells);
// Del:}
// Del:
// Del:
// Del:
// Del:static hash_type * well_hash_copyc(hash_type * well_hash_org)
// Del:{
// Del:  hash_type * well_hash_new = hash_alloc();
// Del:  hash_iter_type * well_iter = hash_iter_alloc( well_hash_org );
// Del:
// Del:  while ( !hash_iter_is_complete(well_iter)) {
// Del:    const char * well = hash_iter_get_next_key( well_iter );
// Del:    hash_type * well_obs_org  = hash_get(well_hash_org, well);
// Del:    hash_type * well_obs_new  = hash_alloc();
// Del:    hash_iter_type * org_iter = hash_iter_alloc( hash_safe_cast(well_obs_org) );
// Del:    
// Del:    while (!hash_iter_is_complete( org_iter )) {
// Del:      const char * key = hash_iter_get_next_key( org_iter );
// Del:      {
// Del:	double obs = hash_get_double(well_obs_org, key);
// Del:	hash_insert_double(well_obs_new, key , obs);
// Del:      }
// Del:    }
// Del:    hash_insert_hash_owned_ref(well_hash_new, well , well_obs_new, hash_free__);
// Del:    hash_iter_free( org_iter );
// Del:  }
// Del:  hash_iter_free( well_iter );
// Del:
// Del:  return well_hash_new;
// Del:}
// Del:
// Del:
// Del:
// Del:
// Del:/**
// Del:   The list of variables WOPR +++ is compiled in - hmmm.
// Del:*/
// Del:
// Del:static hash_type * well_hash_alloc_from_summary(const ecl_sum_type * summary, 
// Del:                                                const stringlist_type * wells , 
// Del:						int restart_nr,
// Del:                                                bool use_h_keywords)
// Del:{
// Del:  hash_type * well_hash = hash_alloc();
// Del:  
// Del:  for(int well_nr = 0; well_nr < stringlist_get_size( wells ); well_nr++)
// Del:  {
// Del:    hash_type * well_obs = hash_alloc();
// Del:
// Del:    // Cleaner than macros.
// Del:    void insert_obs(const char * well_name, const char * obs_name)
// Del:    { 
// Del:      if(ecl_sum_has_well_var(summary, well_name, obs_name)) {
// Del:	int ministep2;
// Del:	double obs;
// Del:
// Del:	ecl_sum_report2ministep_range( summary , restart_nr , NULL , &ministep2);
// Del:        obs = ecl_sum_get_well_var(summary, ministep2, well_name, obs_name);
// Del:        hash_insert_double(well_obs, obs_name, obs);
// Del:
// Del:      } 
// Del:    }
// Del:    
// Del:    void insert_obs_use_h(const char * well_name, const char * obs_name, const char * obs_ins_name)
// Del:      {
// Del:      if(ecl_sum_has_well_var(summary, well_name, obs_name)) {
// Del:	int ministep2;
// Del:	double obs;
// Del:	
// Del:	ecl_sum_report2ministep_range( summary , restart_nr , NULL , &ministep2);
// Del:	obs = ecl_sum_get_well_var(summary, ministep2, well_name, obs_name);
// Del:	hash_insert_double(well_obs, obs_ins_name, obs);
// Del:      }
// Del:    }
// Del:
// Del:    {
// Del:      const char * well = stringlist_iget( wells , well_nr );
// Del:      if(!use_h_keywords)
// Del:	{
// Del:	  insert_obs(well, "WOPR");
// Del:	  insert_obs(well, "WWPR");
// Del:	  insert_obs(well, "WGPR");
// Del:	  insert_obs(well, "WBHP");
// Del:	  insert_obs(well, "WTHP");
// Del:	  insert_obs(well, "WWCT");
// Del:	  insert_obs(well, "WGOR");
// Del:	  insert_obs(well, "WOPT");
// Del:	  insert_obs(well, "WGPT");
// Del:	  insert_obs(well, "WWPT");
// Del:	  insert_obs(well, "WWIR");
// Del:	  insert_obs(well, "WWIT");
// Del:	  insert_obs(well, "WGIR");
// Del:	  insert_obs(well, "WGIT");
// Del:	}
// Del:      else
// Del:	{
// Del:	  insert_obs_use_h(well, "WOPRH", "WOPR");
// Del:	  insert_obs_use_h(well, "WWPRH", "WWPR");
// Del:	  insert_obs_use_h(well, "WGPRH", "WGPR");
// Del:	  insert_obs_use_h(well, "WBHPH", "WBHP");
// Del:	  insert_obs_use_h(well, "WTHPH", "WTHP");
// Del:	  insert_obs_use_h(well, "WWCTH", "WWCT");
// Del:	  insert_obs_use_h(well, "WGORH", "WGOR");
// Del:	  insert_obs_use_h(well, "WOPTH", "WOPT");
// Del:	  insert_obs_use_h(well, "WGPTH", "WGPT");
// Del:	  insert_obs_use_h(well, "WWPTH", "WOPT");
// Del:	  insert_obs_use_h(well, "WWIRH", "WWIR");
// Del:	  insert_obs_use_h(well, "WWITH", "WWIT");
// Del:	  insert_obs_use_h(well, "WGIRH", "WGIR");
// Del:	  insert_obs_use_h(well, "WGITH", "WGIT");
// Del:	}
// Del:      
// Del:      hash_insert_hash_owned_ref(well_hash, well, well_obs, hash_free__);
// Del:    }
// Del:  }
// Del:
// Del:  return well_hash;
// Del:}
// Del:
// Del:
// Del:
// Del:static double well_hash_get_var(hash_type * well_hash, const char * well, const char * var, bool * default_used)
// Del:{
// Del:  if(!hash_has_key(well_hash, well))
// Del:  {
// Del:    *default_used = true;
// Del:    return 0.0;
// Del:  }
// Del:
// Del:  {
// Del:    hash_type * well_obs = hash_get(well_hash, well);
// Del:    if(!hash_has_key(well_obs, var))
// Del:    {
// Del:      *default_used = true;
// Del:      return 0.0;
// Del:    }
// Del:    else
// Del:    {
// Del:      *default_used = false;
// Del:      return  hash_get_double(well_obs, var);
// Del:    }
// Del:  }
// Del:}
// Del:
// Del:
// Del:
// Del:
// Del:/******************************************************************/
// Del:// Functions for manipulating history_node_type.
// Del:
// Del:
// Del:static history_node_type * history_node_alloc_empty()
// Del:{
// Del:  history_node_type * node = util_malloc(sizeof * node, __func__);
// Del:  node->well_hash          = hash_alloc();
// Del:  node->gruptree           = gruptree_alloc();
// Del:  node->restart_nr         = 0;
// Del:  return node;
// Del:}
// Del:
// Del:
// Del:
// Del:static void history_node_free(history_node_type * node)
// Del:{
// Del:  hash_free(node->well_hash);
// Del:  gruptree_free(node->gruptree);
// Del:  free(node);
// Del:}
// Del:
// Del:
// Del:
// Del:static void history_node_free__(void * node)
// Del:{
// Del:  history_node_free( (history_node_type *) node);
// Del:}
// Del:
// Del:
// Del://static void history_node_fwrite(const history_node_type * node, FILE * stream)
// Del://{
// Del://  util_fwrite(&node->node_start_time, sizeof node->node_start_time, 1, stream, __func__);
// Del://  util_fwrite(&node->node_end_time,   sizeof node->node_end_time,   1, stream, __func__);
// Del://  well_hash_fwrite(node->well_hash, stream);
// Del://  gruptree_fwrite(node->gruptree, stream);
// Del://}
// Del://
// Del://
// Del://
// Del://static history_node_type * history_node_fread_alloc(FILE * stream)
// Del://{
// Del://  history_node_type * node = util_malloc(sizeof * node, __func__);
// Del://
// Del://  util_fread(&node->node_start_time, sizeof node->node_start_time, 1, stream, __func__);
// Del://  util_fread(&node->node_end_time,   sizeof node->node_end_time,   1, stream, __func__);
// Del://
// Del://  node->well_hash = well_hash_fread_alloc(stream);
// Del://  node->gruptree  = gruptree_fread_alloc(stream);
// Del://
// Del://  return node;
// Del://}
// Del:
// Del:
// Del:
// Del:/*
// Del:  This function will add the observations from well_hash to the well_hash in
// Del:  the history_node. If the wells are already in the history_node, their data
// Del:  is replaced with the data from well_hash.
// Del:*/
// Del:static void history_node_register_wells(history_node_type * node, hash_type * well_hash)
// Del:{
// Del:  int num_wells = hash_get_size(well_hash);
// Del:  char ** well_list = hash_alloc_keylist(well_hash);
// Del:
// Del:  for(int well_nr = 0; well_nr < num_wells; well_nr++)
// Del:  {
// Del:    if(hash_has_key(node->well_hash, well_list[well_nr]))
// Del:      hash_del(node->well_hash, well_list[well_nr]);
// Del:
// Del:    hash_type * well_obs_ext = hash_get(well_hash, well_list[well_nr]);
// Del:    hash_type * well_obs_int = hash_alloc();
// Del:
// Del:    int num_obs = hash_get_size(well_obs_ext);
// Del:    char ** obs_list = hash_alloc_keylist(well_obs_ext);
// Del:    
// Del:    for(int obs_nr = 0; obs_nr < num_obs; obs_nr++)
// Del:    {
// Del:      double obs = hash_get_double(well_obs_ext, obs_list[obs_nr]);
// Del:      hash_insert_double(well_obs_int, obs_list[obs_nr], obs);
// Del:    }
// Del:    util_free_stringlist(obs_list, num_obs);
// Del:    hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_int, hash_free__);
// Del:  }
// Del:
// Del:  util_free_stringlist(well_list, num_wells);
// Del:}
// Del:
// Del:
// Del:
// Del:static void history_node_delete_wells(history_node_type * node, const char ** well_list, int num_wells)
// Del:{
// Del:  for(int well_nr = 0; well_nr < num_wells; well_nr++)
// Del:  {
// Del:    if(hash_has_key(node->well_hash, well_list[well_nr]))
// Del:    {
// Del:      hash_del(node->well_hash, well_list[well_nr]);
// Del:    }
// Del:  }
// Del:}
// Del:
// Del:
// Del:
// Del:/*
// Del:  This function will add default observations for production wells. This is neccessary, since
// Del:  the well_hash has no concept of well status.
// Del:
// Del:  If some of the wells in well_list are not present, they are created.
// Del:*/
// Del:static void history_node_add_producer_defaults(history_node_type * node, const char ** well_list, int num_wells)
// Del:{
// Del:  for(int well_nr = 0; well_nr < num_wells; well_nr++)
// Del:  {
// Del:    if(hash_has_key(node->well_hash, well_list[well_nr]))
// Del:    {
// Del:      hash_type * well_obs_hash = hash_get(node->well_hash, well_list[well_nr]);
// Del:      hash_insert_double(well_obs_hash, "WWIR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGIR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WOIR", 0.0);
// Del:    }
// Del:    else
// Del:    {
// Del:      hash_type * well_obs_hash = hash_alloc();
// Del:      hash_insert_double(well_obs_hash, "WWIR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGIR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WOIR", 0.0);
// Del:      hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_hash, hash_free__);
// Del:    }
// Del:  }
// Del:}
// Del:
// Del:
// Del:
// Del:/*
// Del:  This function will add default observations for injection wells. This is neccessary, since
// Del:  the well_hash has no concept of well status.
// Del:
// Del:  If some of the wells in well_list are not present, they are created.
// Del:*/
// Del:static void history_node_add_injector_defaults(history_node_type * node, const char ** well_list, int num_wells)
// Del:{
// Del:  for(int well_nr = 0; well_nr < num_wells; well_nr++)
// Del:  {
// Del:    if(hash_has_key(node->well_hash, well_list[well_nr]))
// Del:    {
// Del:      hash_type * well_obs_hash = hash_get(node->well_hash, well_list[well_nr]);
// Del:      hash_insert_double(well_obs_hash, "WOPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWGPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWCT", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGOR", 0.0);
// Del:    }
// Del:    else
// Del:    {
// Del:      hash_type * well_obs_hash = hash_alloc();
// Del:      hash_insert_double(well_obs_hash, "WOPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWGPR", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WWCT", 0.0);
// Del:      hash_insert_double(well_obs_hash, "WGOR", 0.0);
// Del:      hash_insert_hash_owned_ref(node->well_hash, well_list[well_nr], well_obs_hash, hash_free__);
// Del:    }
// Del:  }
// Del:}
// Del:
// Del:
// Del:
// Del:static void history_node_update_gruptree_grups(history_node_type * node, char ** children, char ** parents, int num_pairs)
// Del:{
// Del:  for(int pair = 0; pair < num_pairs; pair++)
// Del:    gruptree_register_grup(node->gruptree, children[pair], parents[pair]);
// Del:}
// Del:
// Del:
// Del:
// Del:static void history_node_update_gruptree_wells(history_node_type * node, char ** children, char ** parents, int num_pairs)
// Del:{
// Del:  for(int pair = 0; pair < num_pairs; pair++)
// Del:    gruptree_register_well(node->gruptree, children[pair], parents[pair]);
// Del:}
// Del:
// Del:
// Del:
// Del:/*
// Del:  The function history_node_parse_data_from_sched_kw updates the
// Del:  history_node_type pointer node with data from sched_kw. I.e., if sched_kw
// Del:  indicates that a producer has been turned into an injector, rate
// Del:  information about the producer shall be removed from node if it exists.
// Del:*/
// Del:static void history_node_parse_data_from_sched_kw(history_node_type * node, const sched_kw_type * sched_kw)
// Del:{
// Del:  switch(sched_kw_get_type(sched_kw))
// Del:  {
// Del:    case(WCONHIST):
// Del:    {
// Del:      hash_type * well_hash = sched_kw_alloc_well_obs_hash(sched_kw);
// Del:      history_node_register_wells(node, well_hash);
// Del:      int num_wells = hash_get_size(well_hash);
// Del:      char ** well_list = hash_alloc_keylist(well_hash);
// Del:      history_node_add_producer_defaults(node, (const char **) well_list, num_wells);
// Del:      util_free_stringlist(well_list, num_wells);
// Del:      hash_free(well_hash);
// Del:      break;
// Del:    }
// Del:    case(WCONPROD):
// Del:    {
// Del:      int num_wells;
// Del:      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
// Del:      history_node_delete_wells(node, (const char **) well_list, num_wells); 
// Del:      history_node_add_producer_defaults(node, (const char **) well_list, num_wells);
// Del:      util_free_stringlist(well_list, num_wells);
// Del:      break;
// Del:    }
// Del:    case(WCONINJ):
// Del:    {
// Del:      int num_wells;
// Del:      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
// Del:      history_node_delete_wells(node, (const char **) well_list, num_wells); 
// Del:      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
// Del:      util_free_stringlist(well_list, num_wells);
// Del:      break;
// Del:    }
// Del:    case(WCONINJE):
// Del:    {
// Del:      int num_wells;
// Del:      char ** well_list = sched_kw_alloc_well_list(sched_kw, &num_wells);
// Del:      history_node_delete_wells(node, (const char **) well_list, num_wells); 
// Del:      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
// Del:      util_free_stringlist(well_list, num_wells);
// Del:      break;
// Del:    }
// Del:    case(WCONINJH):
// Del:    {
// Del:      hash_type * well_hash = sched_kw_alloc_well_obs_hash(sched_kw);
// Del:      history_node_register_wells(node, well_hash);
// Del:      int num_wells = hash_get_size(well_hash);
// Del:      char ** well_list = hash_alloc_keylist(well_hash);
// Del:      history_node_add_injector_defaults(node, (const char **) well_list, num_wells);
// Del:      util_free_stringlist(well_list, num_wells);
// Del:      hash_free(well_hash);
// Del:      break;
// Del:    }
// Del:    case(GRUPTREE):
// Del:    {
// Del:      int num_pairs;
// Del:      char ** children = NULL;
// Del:      char ** parents = NULL;
// Del:      sched_kw_alloc_child_parent_list(sched_kw, &children, &parents, &num_pairs);
// Del:      history_node_update_gruptree_grups(node, children, parents, num_pairs);
// Del:      util_free_stringlist(children, num_pairs);
// Del:      util_free_stringlist(parents, num_pairs);
// Del:      break;
// Del:    }
// Del:    case(WELSPECS):
// Del:    {
// Del:      int num_pairs;
// Del:      char ** children = NULL;
// Del:      char ** parents = NULL;
// Del:      sched_kw_alloc_child_parent_list(sched_kw, &children, &parents, &num_pairs);
// Del:      history_node_update_gruptree_wells(node, children, parents, num_pairs);
// Del:      util_free_stringlist(children, num_pairs);
// Del:      util_free_stringlist(parents, num_pairs);
// Del:      break;
// Del:    }
// Del:    default:
// Del:      break;
// Del:  }
// Del:}
// Del:
// Del:
// Del:
// Del:static history_node_type * history_node_copyc(const history_node_type * node_org)
// Del:{
// Del:  history_node_type * node_new = util_malloc(sizeof * node_new, __func__);
// Del:
// Del:  node_new->node_start_time    = node_org->node_start_time;
// Del:  node_new->node_end_time      = node_org->node_end_time;
// Del:  node_new->restart_nr         = node_org->restart_nr; 
// Del:
// Del:  node_new->well_hash  = well_hash_copyc(node_org->well_hash);
// Del:  node_new->gruptree   = gruptree_copyc(node_org->gruptree);
// Del:
// Del:  return node_new;
// Del:}
// Del:
// Del:
// Del:static void history_node_fprintf(const history_node_type * node, FILE * stream) {
// Del:  fprintf(stream , "%03d:  " , node->restart_nr); util_fprintf_date( node->node_start_time , stream); fprintf(stream , " => "); util_fprintf_date(node->node_end_time , stream); fprintf(stream , "\n");
// Del:}
// Del:
// Del:
// Del:
// Del:/******************************************************************/
// Static functions for manipulating history_type.



static history_type * history_alloc_empty()
{
  history_type * history = util_malloc(sizeof * history, __func__);
  history->ecl_sum       = NULL; 
  return history;
}



//static void history_add_node(history_type * history, history_node_type * node)
//{
//  list_append_list_owned_ref(history->nodes, node, history_node_free__);
//}
//
//
//
//static history_node_type * history_iget_node_ref(const history_type * history, int i)
//{
//  history_node_type * node = list_iget_node_value_ptr(history->nodes, i);
//  return node;
//}



/******************************************************************/
// Exported functions for manipulating history_type. Acess functions further below.


void history_free(history_type * history)
{
  sched_history_free( history->sched_history );
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
history_type * history_alloc_from_sched_file(const char * sep_string , const sched_file_type * sched_file)
{
  history_type * history = history_alloc_empty( );
  history->sched_history = sched_history_alloc( sep_string );
  sched_history_update( history->sched_history , sched_file );
//
//
//
//int num_restart_files = sched_file_get_num_restart_files(sched_file);
//history_node_type * node = NULL;
//for(int block_nr = 0; block_nr < num_restart_files; block_nr++)
//{
//  if(node != NULL) {
//    history_node_type * node_cpy = history_node_copyc(node);
//    node = node_cpy;
//  }
//  else
//    node = history_node_alloc_empty();
//
//  node->node_start_time = sched_file_iget_block_start_time(sched_file, block_nr);
//  node->node_end_time   = sched_file_iget_block_end_time(sched_file, block_nr);
//  node->restart_nr      = block_nr;
//
//  int num_kws = sched_file_iget_block_size(sched_file, block_nr);
//  for(int kw_nr = 0; kw_nr < num_kws; kw_nr++)
//  {
//    sched_kw_type * sched_kw = sched_file_ijget_block_kw_ref(sched_file, block_nr, kw_nr);
//    history_node_parse_data_from_sched_kw(node, sched_kw);
//  }
//  history_add_node(history, node);
//}
//
  history->source = SCHEDULE;
  
  
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

void history_realloc_from_summary(history_type * history, const ecl_sum_type * refcase , bool use_h_keywords) {

  history->ecl_sum = refcase;     /* This function does not really do anthing - it just sets the ecl_sum field of the history instance. */
  if (use_h_keywords)
    history->source = REFCASE_HISTORY;
  else
    history->source = REFCASE_SIMULATED;
  
  
}




/******************************************************************/
// Exported functions for accessing history_type.




/**
  Get the number of restart files the underlying schedule file would produce.
*/
int history_get_num_restarts(const history_type * history)
{
  return sched_history_get_last_history( history->sched_history );
}





/**
  This function takes a key in the same format as the summary.x program, e.g. GOPR:GROUPA, 
  and tries to return the observed value for that key.
*/
double history_get(const history_type * history, int restart_nr, const char * summary_key, bool * default_used)
{
  double value = 0.0;
  
  if (history->source == SCHEDULE) {

    if (sched_history_has_key( history->sched_history , summary_key)) {
      *default_used = false;
      return sched_history_iget( history->sched_history , summary_key , restart_nr);
    } else {
      *default_used = true;
      return 0;
    }
    
      
    /*
    int argc;
    char ** argv;
    
    util_split_string(summary_key, ":", &argc, &argv);
    
    if(argc != 2)
    util_abort("%s: Key \"%s\" does not appear to be a valid summary key.\n", __func__, summary_key);
    
    if(history_str_is_group_name(history, restart_nr, argv[1]))
      value = history_get_group_var(history, restart_nr, argv[1], argv[0], default_used);
    else if(history_str_is_well_name(history, restart_nr, argv[1]))
      value = history_get_well_var(history, restart_nr, argv[1], argv[0], default_used);
    else
      *default_used = true;
    
    util_free_stringlist(argv, argc);
    return value;
    */


  } else {

    /** 100% plain ecl_sum_get... */

    *default_used  = false;  /* Do not have control over this when using the ecl_sum interface. */
    int ministep   = ecl_sum_get_report_ministep_end( history->ecl_sum , restart_nr );
    if (ministep >= 0) {
      char * gen_key = (char *) summary_key;
      if (history->source == REFCASE_HISTORY) {
        /* Must add H to make keywords into history version: */
        const ecl_smspec_type * smspec = ecl_sum_get_smspec( history->ecl_sum );
        const char            * join_string = ecl_smspec_get_join_string( smspec ); 
        
        gen_key = util_alloc_sprintf( "%sH%s%s" , ecl_sum_get_keyword( history->ecl_sum , summary_key ) , join_string , ecl_sum_get_wgname( history->ecl_sum , summary_key ));
      }

      if (ecl_sum_has_general_var(history->ecl_sum , gen_key)) 
        value = ecl_sum_get_general_var( history->ecl_sum , ministep , gen_key );
      else 
        *default_used = true;
        
      if (history->source == REFCASE_HISTORY)
        free( gen_key );
    } else
      *default_used = true;   /* We did not have this ministep. */
    
    return value;
  }
}

//
///**
//  Get the observed value for a well var.
//*/
//double history_get_well_var(const history_type * history, int restart_nr, const char * well, const char * var, bool * default_used)
//{
//  history_node_type * node = history_iget_node_ref(history, restart_nr);
//  return well_hash_get_var(node->well_hash, well, var, default_used);
//}
//
//
//
///**
//  Get the observed value for a group var.
//*/
//double history_get_group_var(const history_type * history, int restart_nr, const char * group, const char * var, bool * default_used)
//{
//  history_node_type * node = history_iget_node_ref(history, restart_nr);
//
//  if(!gruptree_has_grup(node->gruptree, group))
//  {
//    *default_used = true;
//    return 0.0;
//  }
//
//  char * wvar = NULL;
//  if(     strcmp(var, "GOPR") == 0)
//    wvar = "WOPR";
//  else if(strcmp(var, "GWPR") == 0)
//    wvar = "WWPR";
//  else if(strcmp(var, "GGPR") == 0)
//    wvar = "WGPR";
//  else if(strcmp(var, "GOIR") == 0)
//    wvar = "WOIR";
//  else if(strcmp(var, "GWIR") == 0)
//    wvar = "WWIR";
//  else if(strcmp(var, "GGIR") == 0)
//    wvar = "WGIR";
//  else if(strcmp(var, "GOPT") == 0)
//    wvar = "WOPT";
//  else if(strcmp(var, "GWPT") == 0)
//    wvar = "WWPT";
//  else if(strcmp(var, "GGPT") == 0)
//    wvar = "WGPT";
//  else
//  {
//    util_abort("%s: No support for calculating group keyword %s for group %s from well keywords.\n", __func__, var, group);
//  }
//
//  double obs = 0.0;
//  int num_wells;
//  char ** well_list = gruptree_alloc_grup_well_list(node->gruptree, group, &num_wells);
//  *default_used = false;
//  for(int well_nr = 0; well_nr < num_wells; well_nr++)
//  {
//    bool def = false;
//    double obs_inc = well_hash_get_var(node->well_hash, well_list[well_nr], wvar, &def);
//    obs = obs + obs_inc;
//    if(def)
//      *default_used = true;
//  }
//  util_free_stringlist(well_list, num_wells);
//  return obs;
//}

bool history_valid_key( const history_type * history , const char * summary_key ) {
  if (history->source == SCHEDULE) 
    return sched_history_has_key( history->sched_history , summary_key);
  else {
    bool valid;
    char * local_key;
    if (history->source == REFCASE_HISTORY) {
      /* Must create a new key with 'H' for historical values. */
      const ecl_smspec_type * smspec      = ecl_sum_get_smspec( history->ecl_sum );
      const char            * join_string = ecl_smspec_get_join_string( smspec ); 
      
      local_key = util_alloc_sprintf( "%sH%s%s" , ecl_sum_get_keyword( history->ecl_sum , summary_key ) , join_string , ecl_sum_get_wgname( history->ecl_sum , summary_key ));
    } else
      local_key = (char *) summary_key;
    
    valid = ecl_sum_has_general_var( history->ecl_sum , local_key);
    if (history->source == REFCASE_HISTORY) 
      free( local_key );

    return valid;      
  }
}



void history_init_ts( const history_type * history , const char * summary_key , double_vector_type * value, bool_vector_type * valid) {
  double_vector_reset( value );
  bool_vector_reset( valid );
  bool_vector_set_default( valid , false);
  if (history->source == SCHEDULE) {
    for (int tstep = 0; tstep <= sched_history_get_last_history(history->sched_history); tstep++) {
      if (sched_history_open( history->sched_history , summary_key , tstep)) {
        bool_vector_iset( valid , tstep , true );
        double_vector_iset( value , tstep , sched_history_iget( history->sched_history , summary_key , tstep));
      } else
        bool_vector_iset( valid , tstep , false );
    }
  } else {
    char * local_key;
    if (history->source == REFCASE_HISTORY) {
      /* Must create a new key with 'H' for historical values. */
      const ecl_smspec_type * smspec      = ecl_sum_get_smspec( history->ecl_sum );
      const char            * join_string = ecl_smspec_get_join_string( smspec ); 
        
      local_key = util_alloc_sprintf( "%sH%s%s" , ecl_sum_get_keyword( history->ecl_sum , summary_key ) , join_string , ecl_sum_get_wgname( history->ecl_sum , summary_key ));
    } else
      local_key = (char *) summary_key;
  
    {
      for (int tstep = 0; tstep <= sched_history_get_last_history(history->sched_history); tstep++) {
        int ministep   = ecl_sum_get_report_ministep_end( history->ecl_sum , tstep );
        if (ministep >= 0) {
          double_vector_iset( value , tstep , ecl_sum_get_general_var( history->ecl_sum , ministep , local_key ));
          bool_vector_iset( valid , tstep , true );
        } else
          bool_vector_iset( valid , tstep , false );    /* Did not have this report step */
      }
    }
    
    if (history->source == REFCASE_HISTORY) 
      free( local_key );
  }
}


/**
  This function alloc's two vectors of length num_restarts from a summary key.
  In the vector value, the observed values or default values for each restart
  is stored. If the default value is used, the corresponding element in the
  vector default_used will be true.
*/
//void   history_alloc_time_series_from_summary_key
//(
//                    const history_type * history,
//                    const char         * summary_key,
//                    double            ** __value,
//                    bool              ** __default_used
//)
//{
//  int num_restarts = history_get_num_restarts(history);
//  
//  double * value        = util_malloc(num_restarts * sizeof * value ,        __func__);
//  bool   * default_used = util_malloc(num_restarts * sizeof * default_used , __func__);
//
//  int argc;
//  char ** argv;
//  util_split_string(summary_key, ":", &argc, &argv);
//  if(argc != 2)
//    util_abort("%s: Key \"%s\" does not appear to be a valid summary key.\n", __func__, summary_key);
//
//  for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++)
//  {
//
//    value[restart_nr] = history_get( history , restart_nr , summary_key , &default_used[restart_nr]);
//  }
//
//  util_free_stringlist(argv, argc);
//  *__value        = value;
//  *__default_used = default_used;
//}



//time_t history_iget_node_start_time(const history_type * history, int node_nr)
//{
//  history_node_type * history_node = history_iget_node_ref(history, node_nr);
//  return history_node->node_start_time;
//}
//
//
//
//time_t history_iget_node_end_time(const history_type * history, int node_nr)
//{
//  history_node_type * history_node = history_iget_node_ref(history, node_nr);
//  return history_node->node_end_time;
//}



//int history_get_restart_nr_from_time_t(const history_type * history, time_t time)
//{
//  int num_restart_files = history_get_num_restarts(history);
//  int restart_nr        = 0;
//
//  while (restart_nr < num_restart_files) {
//    time_t node_end_time = history_iget_node_end_time(history, restart_nr);
//    
//    if (node_end_time == time)  
//      break;                    /* Got it */
//    else
//      restart_nr++;
//  }
//  
//  if (restart_nr == num_restart_files) {
//    int sec,min,hour;
//    int mday,year,month;
//    
//    util_set_datetime_values(time , &sec , &min , &hour , &mday , &month , &year);
//    util_abort("%s: Time variable:%02d/%02d/%4d  %02d:%02d:%02d   does not cooincide with any restart file. Aborting.\n", __func__ , mday,month,year,hour,min,sec);
//  }
//  return restart_nr;
//}



//int history_get_restart_nr_from_days(const history_type * history, double days)
//{
//  if(days < 0.0)
//    util_abort("%s: Cannot find a restart nr from a negative production period (days was %g).\n",
//               __func__, days);
//
//  time_t time = history_iget_node_start_time(history, 0);
//  util_inplace_forward_days(&time, days);
//
//  return history_get_restart_nr_from_time_t(history, time);
//}


/* Uncertain about the first node - offset problems +++ ?? 
   Changed to use node_end_time() at svn ~ 2850

   Changed to sched_history at svn ~2940
*/
time_t history_get_time_t_from_restart_nr( const history_type * history , int restart_nr) {
  return sched_history_iget_time_t( history->sched_history , restart_nr);
  //return history_iget_node_end_time(history , restart_nr);
}




