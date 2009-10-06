#include <stdlib.h>
#include <string.h>
#include <vector.h>
#include <util.h>
#include <sched_kw_wconinjh.h>
#include <sched_util.h>
#include <hash.h>
#include <stringlist.h>

/*
  Define the maximum number of keywords in a WCONINJH record.
*/

#define WCONINJH_NUM_KW 8
#define ECL_DEFAULT_KW "*"



typedef enum {WATER, GAS, OIL} inj_flag_type;
#define INJ_WATER_STRING "WATER"
#define INJ_GAS_STRING   "GAS"
#define INJ_OIL_STRING   "OIL"



typedef enum {OPEN, STOP, SHUT} st_flag_type;
#define ST_OPEN_STRING "OPEN"
#define ST_STOP_STRING "STOP"
#define ST_SHUT_STRING "SHUT"

struct sched_kw_wconhist_struct{
  vector_type * wells;
};



typedef struct wconinjh_well_struct wconinjh_well_type;



struct wconinjh_well_struct{
  /*
    def: Read as defaulted, not defined!
  */
  bool          def[WCONINJH_NUM_KW];

  char          * name;
  inj_flag_type   inj_phase;
  st_flag_type    status;
  double          inj_rate;
  double          bhp;
  double          thp;
  int             vfptable;
  double          vapdiscon;
};



static char * get_inj_string_from_flag(inj_flag_type inj_phase)
{
  switch(inj_phase)
  {
    case(WATER):
      return INJ_WATER_STRING;
    case(GAS):
      return INJ_GAS_STRING;
    case(OIL):
      return INJ_OIL_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
}



static char * get_st_string_from_flag(st_flag_type status)
{
  switch(status)
  {
    case(OPEN):
      return ST_OPEN_STRING;
    case(STOP):
      return ST_STOP_STRING;
    case(SHUT):
      return ST_SHUT_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
}


/*
  No default defined
*/
static inj_flag_type get_inj_flag_from_string(const char * inj_phase)
{
  if(      strcmp(inj_phase, INJ_WATER_STRING) == 0)
    return WATER;
  else if( strcmp(inj_phase, INJ_GAS_STRING)   == 0)
    return GAS;
  else if( strcmp(inj_phase, INJ_OIL_STRING)   == 0)
    return OIL;
  else
  {
    util_abort("%s: Couldn't recognize %s as a injection phase.\n", __func__, inj_phase);
    return 0;
  }
}



static st_flag_type get_st_flag_from_string(const char * status)
{
  if(      strcmp(status, ST_OPEN_STRING) == 0)
    return OPEN;
  else if( strcmp(status, ST_STOP_STRING) == 0)
    return STOP;
  else if( strcmp(status, ST_SHUT_STRING) == 0)
    return SHUT;
  else
  {
    util_abort("%s: Could'nt recognize %s as a well status.\n", __func__, status);
    return 0;
  }
}



static wconinjh_well_type * wconinjh_well_alloc_empty()
{
  wconinjh_well_type * well = util_malloc(sizeof * well, __func__);
  well->name = NULL;
  return well;
}



static void wconinjh_well_free(wconinjh_well_type * well)
{
  free(well->name);
  free(well);
}



static void wconinjh_well_free__(void * well)
{
  wconinjh_well_free( (wconinjh_well_type *) well);
}



static void wconinjh_well_fprintf(const wconinjh_well_type * well, FILE * stream)
{
  fprintf(stream, "  ");
  sched_util_fprintf_qst(well->def[0], well->name                               , 8, stream);
  sched_util_fprintf_qst(well->def[1], get_inj_string_from_flag(well->inj_phase), 5, stream);
  sched_util_fprintf_qst(well->def[2], get_st_string_from_flag(well->status)    , 4, stream);
  sched_util_fprintf_dbl(well->def[3], well->inj_rate                           , 9 , 3, stream);
  sched_util_fprintf_dbl(well->def[4], well->bhp                                , 9 , 3, stream);
  sched_util_fprintf_dbl(well->def[5], well->thp                                , 9 , 3, stream);
  sched_util_fprintf_int(well->def[6], well->vfptable                           , 4, stream);
  sched_util_fprintf_dbl(well->def[7], well->vapdiscon                          , 9 , 3, stream);
  fprintf(stream, "/\n");
}









static wconinjh_well_type * wconinjh_well_alloc_from_tokens(const stringlist_type * line_tokens ) {

  wconinjh_well_type * well = wconinjh_well_alloc_empty();
  sched_util_init_default( line_tokens , well->def );
  
  well->name      = util_alloc_string_copy(stringlist_iget(line_tokens , 0));
  well->inj_phase = get_inj_flag_from_string(stringlist_iget(line_tokens , 1));
  well->status    = get_st_flag_from_string(stringlist_iget(line_tokens , 2));
  well->inj_rate  = sched_util_atof(stringlist_iget(line_tokens , 3));
  well->bhp       = sched_util_atof(stringlist_iget(line_tokens , 4));
  well->thp       = sched_util_atof(stringlist_iget(line_tokens , 5));
  well->vfptable  = sched_util_atoi(stringlist_iget(line_tokens , 6));
  well->vapdiscon = sched_util_atof(stringlist_iget(line_tokens , 7));
  
  return well;
}



static hash_type * wconinjh_well_export_obs_hash(const wconinjh_well_type * well) {
  hash_type * obs_hash = hash_alloc();

  if(!well->def[3])
  {
    switch(well->inj_phase)
    {
      case(WATER):
        hash_insert_double(obs_hash, "WWIR", well->inj_rate);
        break;
      case(GAS):
        hash_insert_double(obs_hash, "WGIR", well->inj_rate);
        break;
      case(OIL):
        hash_insert_double(obs_hash, "WOIR", well->inj_rate);
        break;
      default:
        break;
    }
  }
  if(!well->def[4])
    hash_insert_double(obs_hash, "WBHP", well->bhp);
  if(!well->def[5])
    hash_insert_double(obs_hash, "WTHP", well->thp);

  return obs_hash;
}


static void sched_kw_wconinjh_add_well( sched_kw_wconinjh_type * kw , wconinjh_well_type * well) {
  vector_append_owned_ref(kw->wells , well , wconinjh_well_free__);
}



static sched_kw_wconinjh_type * sched_kw_wconinjh_alloc_empty()
{
  sched_kw_wconinjh_type * kw = util_malloc(sizeof * kw, __func__);
  kw->wells = vector_alloc_new();
  return kw;
}



/***********************************************************************/


sched_kw_wconinjh_type * sched_kw_wconinjh_alloc(const stringlist_type * tokens , int * token_index ) {
  sched_kw_wconinjh_type * kw = sched_kw_wconinjh_alloc_empty();
  int eokw                    = false;
  do {
    stringlist_type * line_tokens = sched_util_alloc_line_tokens( tokens , false, WCONINJH_NUM_KW , token_index );
    if (line_tokens == NULL)
      eokw = true;
    else {
      wconinjh_well_type * well = wconinjh_well_alloc_from_tokens( line_tokens );
      sched_kw_wconinjh_add_well( kw , well );
      stringlist_free( line_tokens );
    } 
  } while (!eokw);
  return kw;  
}


void sched_kw_wconinjh_free(sched_kw_wconinjh_type * kw)
{
  vector_free(kw->wells);
  free(kw);
}



void sched_kw_wconinjh_fprintf(const sched_kw_wconinjh_type * kw, FILE * stream)
{
  int size = vector_get_size(kw->wells);
    
  fprintf(stream, "WCONINJH\n");
  for(int i=0; i<size; i++)
  {
    const wconinjh_well_type * well = vector_iget_const( kw->wells, i );
    wconinjh_well_fprintf(well, stream);
  }
  fprintf(stream,"/\n\n");
}




/***********************************************************************/



hash_type * sched_kw_wconinjh_alloc_well_obs_hash(const sched_kw_wconinjh_type * kw)
{
  hash_type * well_hash = hash_alloc();

  int num_wells = vector_get_size(kw->wells);
  
  for(int well_nr=0; well_nr<num_wells; well_nr++)
  {
    const wconinjh_well_type * well = vector_iget_const(kw->wells, well_nr);
    hash_type * obs_hash = wconinjh_well_export_obs_hash(well);
    hash_insert_hash_owned_ref(well_hash, well->name, obs_hash, hash_free__);
  }

  return well_hash;
}

sched_kw_wconinjh_type * sched_kw_wconinjh_copyc(const sched_kw_wconinjh_type * kw) {
  util_abort("%s: not implemented ... \n",__func__);
  return NULL;
}



/***********************************************************************/
KW_IMPL(wconinjh)
