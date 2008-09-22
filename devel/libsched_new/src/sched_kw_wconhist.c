#include <string.h>
#include <list.h>
#include <util.h>
#include <sched_kw_wconhist.h>
#include <sched_util.h>

/*
  Define the maximum number of keywords in a WCONHIST record.
  Note that this includes wet gas rate, which is only supported
  by ECL 300.
*/

#define WCONHIST_NUM_KW 11
#define ECL_DEFAULT_KW "*"


typedef enum {OPEN, STOP, SHUT} st_flag_type;
#define ST_OPEN_STRING "OPEN"
#define ST_STOP_STRING "STOP"
#define ST_SHUT_STRING "SHUT"

typedef enum {ORAT, WRAT, GRAT, LRAT, RESV} cm_flag_type;
#define CM_ORAT_STRING "ORAT"
#define CM_WRAT_STRING "WRAT"
#define CM_GRAT_STRING "GRAT"
#define CM_LRAT_STRING "LRAT"
#define CM_RESV_STRING "RESV"


struct sched_kw_wconhist_struct{
  list_type * wells;
};



typedef struct wconhist_well_struct wconhist_well_type;



struct wconhist_well_struct{
  /*
    def: Read as defaulted, not defined!
  */
  bool          def[WCONHIST_NUM_KW];

  char        * name;
  st_flag_type  status;
  cm_flag_type  cmode;
  double        orat;
  double        wrat;
  double        grat;
  int           vfptable;
  double        alift;
  double        thp;
  double        bhp;
  double        wgrat;
};



static char * get_st_string(st_flag_type status)
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



static char * get_cm_string(cm_flag_type cmode)
{
  switch(cmode)
  {
    case(ORAT):
      return CM_ORAT_STRING;
    case(WRAT):
      return CM_WRAT_STRING;
    case(GRAT):
      return CM_GRAT_STRING;
    case(LRAT):
      return CM_LRAT_STRING;
    case(RESV):
      return CM_RESV_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
}



static st_flag_type get_st_flag_from_string(const char * st_string)
{
  if( strcmp(st_string, ST_OPEN_STRING) == 0)
    return OPEN; 
  else if( strcmp(st_string, ST_STOP_STRING) == 0)
    return STOP; 
  else if( strcmp(st_string, ST_SHUT_STRING) == 0)
    return SHUT; 
  else
  {
    util_abort("%s: Could not recognize %s as a well status.\n", __func__, st_string);
    return 0;
  }
}



static cm_flag_type get_cm_flag_from_string(const char * cm_string)
{
  if(     strcmp(cm_string, CM_ORAT_STRING) == 0)
    return ORAT;
  else if(strcmp(cm_string, CM_WRAT_STRING) == 0)
    return WRAT;
  else if(strcmp(cm_string, CM_GRAT_STRING) == 0)
    return GRAT;
  else if(strcmp(cm_string, CM_LRAT_STRING) == 0)
    return LRAT;
  else if(strcmp(cm_string, CM_RESV_STRING) == 0)
    return RESV;
  else
  {
    util_abort("%s: Could not recognize %s as a control mode.\n", __func__, cm_string);
    return 0;
  }
}




static wconhist_well_type * wconhist_well_alloc_empty()
{
  wconhist_well_type * well = util_malloc(sizeof * well, __func__);
  well->name = NULL;
  return well;
}



static void wconhist_well_free(wconhist_well_type * well)
{
  free(well->name);
  free(well);
}



static void wconhist_well_free__(void * well)
{
  wconhist_well_free( (wconhist_well_type *) well);
}



static void wconhist_well_fprintf(const wconhist_well_type * well, FILE * stream)
{
  fprintf(stream, "  ");
  sched_util_fprintf_qst(well->def[0],  well->name                 , 8,    stream);
  sched_util_fprintf_qst(well->def[1],  get_st_string(well->status), 4,    stream);
  sched_util_fprintf_qst(well->def[2],  get_cm_string(well->cmode) , 4,    stream);
  sched_util_fprintf_dbl(well->def[3],  well->orat                 , 9, 3, stream);
  sched_util_fprintf_dbl(well->def[4],  well->wrat                 , 9, 3, stream);
  sched_util_fprintf_dbl(well->def[5],  well->grat                 , 9, 3, stream);
  sched_util_fprintf_int(well->def[6],  well->vfptable             , 4,    stream);
  sched_util_fprintf_dbl(well->def[7],  well->alift                , 9, 3, stream);
  sched_util_fprintf_dbl(well->def[8],  well->thp                  , 9, 3, stream);
  sched_util_fprintf_dbl(well->def[9],  well->bhp                  , 9, 3, stream);
  sched_util_fprintf_dbl(well->def[10], well->wgrat                , 9, 3, stream);
  fprintf(stream, "/\n");
}



static void wconhist_well_fwrite(const wconhist_well_type * well, FILE * stream)
{
  util_fwrite_string(well->name, stream);

  util_fwrite(&well->status,   sizeof well->status   , 1, stream, __func__);
  util_fwrite(&well->cmode,    sizeof well->cmode    , 1, stream, __func__);
  util_fwrite(&well->orat,     sizeof well->orat     , 1, stream, __func__);
  util_fwrite(&well->wrat,     sizeof well->wrat     , 1, stream, __func__);
  util_fwrite(&well->grat,     sizeof well->grat     , 1, stream, __func__);
  util_fwrite(&well->vfptable, sizeof well->vfptable , 1, stream, __func__);
  util_fwrite(&well->alift,    sizeof well->alift    , 1, stream, __func__);
  util_fwrite(&well->thp,      sizeof well->thp      , 1, stream, __func__);
  util_fwrite(&well->bhp,      sizeof well->bhp      , 1, stream, __func__);
  util_fwrite(&well->wgrat,    sizeof well->wgrat    , 1, stream, __func__);

  util_fwrite(&well->def,      sizeof well->def[0]   , WCONHIST_NUM_KW, stream, __func__);
}



static wconhist_well_type * wconhist_well_fread_alloc(FILE * stream)
{
  wconhist_well_type * well = wconhist_well_alloc_empty();

  well->name = util_fread_alloc_string(stream);

  util_fread(&well->status,   sizeof well->status   , 1, stream, __func__);
  util_fread(&well->cmode,    sizeof well->cmode    , 1, stream, __func__);
  util_fread(&well->orat,     sizeof well->orat     , 1, stream, __func__);
  util_fread(&well->wrat,     sizeof well->wrat     , 1, stream, __func__);
  util_fread(&well->grat,     sizeof well->grat     , 1, stream, __func__);
  util_fread(&well->vfptable, sizeof well->vfptable , 1, stream, __func__);
  util_fread(&well->alift,    sizeof well->alift    , 1, stream, __func__);
  util_fread(&well->thp,      sizeof well->thp      , 1, stream, __func__);
  util_fread(&well->bhp,      sizeof well->bhp      , 1, stream, __func__);
  util_fread(&well->wgrat,    sizeof well->wgrat    , 1, stream, __func__);

  util_fread(&well->def,      sizeof well->def[0]   , WCONHIST_NUM_KW, stream, __func__);

  return well;
}



static wconhist_well_type * wconhist_well_alloc_from_string(char ** token_list)
{
  wconhist_well_type * well = wconhist_well_alloc_empty();

  {
    for(int i=0; i<WCONHIST_NUM_KW; i++)
    {
      if(token_list[i] == NULL)
        well->def[i] = true;
      else
        well->def[i] = false;
    }
  }

  well->name  = util_alloc_string_copy(token_list[0]);

  if(!well->def[1])
    well->status = get_st_flag_from_string(token_list[1]);
  if(!well->def[2])
    well->cmode = get_cm_flag_from_string(token_list[2]);
  if(!well->def[3])
    well->orat = sched_util_atof(token_list[3]); 
  if(!well->def[4])
    well->wrat = sched_util_atof(token_list[4]); 
  if(!well->def[5])
    well->grat = sched_util_atof(token_list[5]); 
  if(!well->def[6])
    well->vfptable = sched_util_atoi(token_list[6]);
  if(!well->def[7])
    well->alift = sched_util_atof(token_list[7]);
  if(!well->def[8])
    well->thp = sched_util_atof(token_list[8]);
  if(!well->def[9])
    well->bhp = sched_util_atof(token_list[9]);
  if(!well->def[10])
    well->wgrat = sched_util_atof(token_list[10]);

  return well;
}



static hash_type * wconhist_well_export_obs_hash(const wconhist_well_type * well)
{
  hash_type * obs_hash = hash_alloc();

  if(!well->def[3])
    hash_insert_double(obs_hash, "WOPR", well->orat);
  if(!well->def[4])
    hash_insert_double(obs_hash, "WWPR", well->wrat);
  if(!well->def[5])
    hash_insert_double(obs_hash, "WGPR", well->grat);
  if(!well->def[8])
    hash_insert_double(obs_hash, "WTHP", well->thp);
  if(!well->def[9])
    hash_insert_double(obs_hash, "WBHP", well->bhp);
  if(!well->def[10])
    hash_insert_double(obs_hash, "WWGPR", well->wgrat);

  // Water cut. Is this the correct definition?!
  if(!well->def[3] && !well->def[4])
  {
    double wct;
    if(well->orat + well->wrat > 0.0)
      wct = well->wrat / (well->orat + well->wrat);
    else
      wct = 0.0;

    hash_insert_double(obs_hash, "WWCT", wct);
  }

  // Gas oil ratio.
  if(!well->def[3] && !well->def[5])
  {
    double gor;
    if(well->orat > 0.0)
    {
      gor = well->grat / well->orat;
      hash_insert_double(obs_hash, "WGOR", gor);
    }
  }

  return obs_hash;
}



static void sched_kw_wconhist_add_line(sched_kw_wconhist_type * kw, const char *line)
{
  int tokens;
  char ** token_list;
  wconhist_well_type * well;

  sched_util_parse_line(line, &tokens, &token_list, WCONHIST_NUM_KW, NULL);

  well = wconhist_well_alloc_from_string(token_list);
  list_append_list_owned_ref(kw->wells, well, wconhist_well_free__);

  util_free_stringlist(token_list, tokens);
}


static sched_kw_wconhist_type * sched_kw_wconhist_alloc()
{
  sched_kw_wconhist_type * kw = util_malloc(sizeof * kw, __func__);
  kw->wells = list_alloc();
  return kw;
}



/***********************************************************************/



sched_kw_wconhist_type * sched_kw_wconhist_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_wconhist_type * kw = sched_kw_wconhist_alloc();

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before WCONHIST was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_wconhist_add_line(kw, line);
      free(line);
    }
  }
  return kw;
}



void sched_kw_wconhist_free(sched_kw_wconhist_type * kw)
{
  list_free(kw->wells);
  free(kw);
}



void sched_kw_wconhist_fprintf(const sched_kw_wconhist_type * kw, FILE * stream)
{
  int size = list_get_size(kw->wells);

  fprintf(stream, "WCONHIST\n");
  for(int i=0; i<size; i++)
  {
    wconhist_well_type * well = list_iget_node_value_ptr(kw->wells, i);
    wconhist_well_fprintf(well, stream);
  }
  fprintf(stream,"/\n\n");
}



void sched_kw_wconhist_fwrite(const sched_kw_wconhist_type * kw, FILE * stream)
{
  int size = list_get_size(kw->wells);
  util_fwrite(&size, sizeof size, 1, stream, __func__);
  for(int i=0; i<size; i++)
  {
    wconhist_well_type * well = list_iget_node_value_ptr(kw->wells, i);
    wconhist_well_fwrite(well, stream);
  }
}


sched_kw_wconhist_type * sched_kw_wconhist_fread_alloc(FILE * stream)
{
  int size;
  sched_kw_wconhist_type * kw = sched_kw_wconhist_alloc();
  util_fread(&size, sizeof size, 1, stream, __func__);

  for(int i=0; i<size; i++)
  {
    wconhist_well_type * well = wconhist_well_fread_alloc(stream);
    list_append_list_owned_ref(kw->wells, well, wconhist_well_free__);
  }

  return kw;
}



/***********************************************************************/



hash_type * sched_kw_wconhist_alloc_well_obs_hash(const sched_kw_wconhist_type * kw)
{
  hash_type * well_hash = hash_alloc();

  int num_wells = list_get_size(kw->wells);
  
  for(int well_nr=0; well_nr<num_wells; well_nr++)
  {
    wconhist_well_type * well = list_iget_node_value_ptr(kw->wells, well_nr);
    hash_type * obs_hash = wconhist_well_export_obs_hash(well);
    hash_insert_hash_owned_ref(well_hash, well->name, obs_hash, hash_free__);
  }

  return well_hash;
}



/***********************************************************************/

KW_FSCANF_ALLOC_IMPL(wconhist)
KW_FWRITE_IMPL(wconhist)
KW_FREAD_ALLOC_IMPL(wconhist)
KW_FREE_IMPL(wconhist)
KW_FPRINTF_IMPL(wconhist)
