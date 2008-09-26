#include <string.h>
#include <list.h>
#include <util.h>
#include <sched_kw_wconinjh.h>
#include <sched_util.h>

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
  list_type * wells;
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



static inj_flag_type get_inj_flag_from_string(char * inj_phase)
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



static st_flag_type get_st_flag_from_string(char * status)
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



static void wconinjh_well_fwrite(const wconinjh_well_type * well, FILE * stream)
{
  util_fwrite_string(well->name, stream);
  
  util_fwrite(&well->inj_phase, sizeof well->inj_phase, 1, stream, __func__);
  util_fwrite(&well->status   , sizeof well->status   , 1, stream, __func__);
  util_fwrite(&well->inj_rate , sizeof well->inj_rate , 1, stream, __func__);
  util_fwrite(&well->bhp      , sizeof well->bhp      , 1, stream, __func__);
  util_fwrite(&well->thp      , sizeof well->thp      , 1, stream, __func__);
  util_fwrite(&well->vfptable , sizeof well->vfptable , 1, stream, __func__);
  util_fwrite(&well->vapdiscon, sizeof well->vapdiscon, 1, stream, __func__);

  util_fwrite(&well->def, sizeof well->def[0], WCONINJH_NUM_KW, stream, __func__);
}



static wconinjh_well_type * wconinjh_well_fread_alloc(FILE * stream)
{
  wconinjh_well_type * well = wconinjh_well_alloc_empty();

  well->name = util_fread_alloc_string( stream);
  
  util_fread(&well->inj_phase, sizeof well->inj_phase, 1, stream, __func__);
  util_fread(&well->status   , sizeof well->status   , 1, stream, __func__);
  util_fread(&well->inj_rate , sizeof well->inj_rate , 1, stream, __func__);
  util_fread(&well->bhp      , sizeof well->bhp      , 1, stream, __func__);
  util_fread(&well->thp      , sizeof well->thp      , 1, stream, __func__);
  util_fread(&well->vfptable , sizeof well->vfptable , 1, stream, __func__);
  util_fread(&well->vapdiscon, sizeof well->vapdiscon, 1, stream, __func__);

  util_fread(&well->def, sizeof well->def[0], WCONINJH_NUM_KW, stream, __func__);

  return well;
}



static wconinjh_well_type * wconinjh_well_alloc_from_string(char ** token_list)
{
  wconinjh_well_type * well = wconinjh_well_alloc_empty();

  {
    for(int i=0; i<WCONINJH_NUM_KW; i++)
    {
      if(token_list[i] == NULL)
        well->def[i] = true;
      else
        well->def[i] = false;
    }
  }

  well->name  = util_alloc_string_copy(token_list[0]);

  if(!well->def[1])
    well->inj_phase = get_inj_flag_from_string(token_list[1]);
  if(!well->def[2])
    well->status = get_st_flag_from_string(token_list[2]);
  if(!well->def[3])
    well->inj_rate = sched_util_atof(token_list[3]);
  if(!well->def[4])
    well->bhp = sched_util_atof(token_list[4]);
  if(!well->def[5])
    well->thp = sched_util_atof(token_list[5]);
  if(!well->def[6])
    well->vfptable = sched_util_atoi(token_list[6]);
  if(!well->def[7])
    well->vapdiscon = sched_util_atof(token_list[7]);

  return well;
}



static hash_type * wconinjh_well_export_obs_hash(const wconinjh_well_type * well)
{
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



static void sched_kw_wconinjh_add_line(sched_kw_wconinjh_type * kw, const char *line)
{
  int tokens;
  char ** token_list;
  wconinjh_well_type * well;

  sched_util_parse_line(line, &tokens, &token_list, WCONINJH_NUM_KW, NULL);

  well = wconinjh_well_alloc_from_string(token_list);
  list_append_list_owned_ref(kw->wells, well, wconinjh_well_free__);

  util_free_stringlist(token_list, tokens);
}



static sched_kw_wconinjh_type * sched_kw_wconinjh_alloc()
{
  sched_kw_wconinjh_type * kw = util_malloc(sizeof * kw, __func__);
  kw->wells = list_alloc();
  return kw;
}



/***********************************************************************/


sched_kw_wconinjh_type * sched_kw_wconinjh_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_wconinjh_type * kw = sched_kw_wconinjh_alloc();

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before WCONINJH was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_wconinjh_add_line(kw, line);
      free(line);
    }
  }
  return kw;
}



void sched_kw_wconinjh_free(sched_kw_wconinjh_type * kw)
{
  list_free(kw->wells);
  free(kw);
}



void sched_kw_wconinjh_fprintf(const sched_kw_wconinjh_type * kw, FILE * stream)
{
  int size = list_get_size(kw->wells);

  fprintf(stream, "WCONINJH\n");
  for(int i=0; i<size; i++)
  {
    wconinjh_well_type * well = list_iget_node_value_ptr(kw->wells, i);
    wconinjh_well_fprintf(well, stream);
  }
  fprintf(stream,"/\n\n");
}



void sched_kw_wconinjh_fwrite(const sched_kw_wconinjh_type * kw, FILE * stream)
{
  int size = list_get_size(kw->wells);
  util_fwrite(&size, sizeof size, 1, stream, __func__);
  for(int i=0; i<size; i++)
  {
    wconinjh_well_type * well = list_iget_node_value_ptr(kw->wells, i);
    wconinjh_well_fwrite(well, stream);
  }
}


sched_kw_wconinjh_type * sched_kw_wconinjh_fread_alloc(FILE * stream)
{
  int size;
  sched_kw_wconinjh_type * kw = sched_kw_wconinjh_alloc();
  util_fread(&size, sizeof size, 1, stream, __func__);

  for(int i=0; i<size; i++)
  {
    wconinjh_well_type * well = wconinjh_well_fread_alloc(stream);
    list_append_list_owned_ref(kw->wells, well, wconinjh_well_free__);
  }

  return kw;
}



/***********************************************************************/



hash_type * sched_kw_wconinjh_alloc_well_obs_hash(const sched_kw_wconinjh_type * kw)
{
  hash_type * well_hash = hash_alloc();

  int num_wells = list_get_size(kw->wells);
  
  for(int well_nr=0; well_nr<num_wells; well_nr++)
  {
    wconinjh_well_type * well = list_iget_node_value_ptr(kw->wells, well_nr);
    hash_type * obs_hash = wconinjh_well_export_obs_hash(well);
    hash_insert_hash_owned_ref(well_hash, well->name, obs_hash, hash_free__);
  }

  return well_hash;
}


/***********************************************************************/

KW_FSCANF_ALLOC_IMPL(wconinjh)
KW_FWRITE_IMPL(wconinjh)
KW_FREAD_ALLOC_IMPL(wconinjh)
KW_FREE_IMPL(wconinjh)
KW_FPRINTF_IMPL(wconinjh)
