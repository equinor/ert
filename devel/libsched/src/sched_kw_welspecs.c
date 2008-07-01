#include <sched_kw_welspecs.h>
#include <list.h>
#include <util.h>
#include <sched_util.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

struct sched_kw_welspecs_struct
{
  list_type * welspec_list;
};

/*
  See ECLIPSE Reference Manual, section WELSPECS
  for an explantion of the members in the 
  welspec_type struct.
*/

typedef enum {PH_OIL, PH_WAT, PH_GAS, PH_LIQ} phase_type;
#define PH_OIL_STRING "OIL"
#define PH_WAT_STRING "WATER"
#define PH_GAS_STRING "GAS"
#define PH_LIQ_STRING "LIQ"



typedef enum {IE_STD, IE_NO, IE_RG, IE_YES, IE_PP, IE_GPP} inflow_eq_type;
#define IE_STD_STRING "STD"
#define IE_NO_STRING  "NO"
#define IE_RG_STRING  "R-G"
#define IE_YES_STRING "YES"
#define IE_PP_STRING  "P-P"
#define IE_GPP_STRING "GPP"



typedef enum {AS_STOP, AS_SHUT} auto_shut_type;
#define AS_STOP_STRING "STOP"
#define AS_SHUT_STRING "SHUT"



typedef enum {CF_YES, CF_NO} crossflow_type; 
#define CF_YES_STRING "YES"
#define CF_NO_STRING  "NO"

typedef enum {HD_SEG,  HD_AVG} hdstat_head_type;
#define HD_SEG_STRING "SEG"
#define HD_AVG_STRING "AVG"

typedef struct
{
  /*
    def : Read as default, not as defined.
  */
  bool           * def;

  char           * name;
  char           * group;
  int              hh_i;
  int              hh_j;
  double           md;
  phase_type       phase;
  double           drain_rad;
  inflow_eq_type   inflow_eq;
  auto_shut_type   auto_shut;
  crossflow_type   crossflow;
  int              pvt_region;
  hdstat_head_type hdstat_head;
  int              fip_region;
  char           * fs_kw1;
  char           * fs_kw2;
  char           * ecl300_kw;

} welspec_type;



static char * get_phase_string(phase_type * phase)
{
  switch(*phase)
  {
    case(PH_OIL):
      return PH_OIL_STRING;
    case(PH_WAT):
      return PH_WAT_STRING;
    case(PH_GAS):
      return PH_GAS_STRING;
    case(PH_LIQ):
      return PH_LIQ_STRING;
    default:
      util_abort("%s: Internal error - aborting.\n",__func__);
  }
};



static char * get_inflow_eq_string(inflow_eq_type *eq)
{
  switch(*eq)
  {
  }
};



static void welspec_sched_fprintf(const welspec_type * ws, FILE * stream)
{
  fprintf(stream, " ");
  sched_util_fprintf_qst(ws->def[0]           , ws->name           , 8,     stream);
  sched_util_fprintf_qst(ws->def[1]           , ws->group          , 8,     stream);
  sched_util_fprintf_int(ws->def[2]           , ws->hh_i           , 4,     stream);
  sched_util_fprintf_int(ws->def[3]           , ws->hh_j           , 4,     stream);

  sched_util_fprintf_dbl(ws->def[4]           , ws->md             , 9, 3,  stream);
};



static welspec_type * welspec_alloc_empty()
{
  welspec_type *ws = util_malloc(sizeof *ws,__func__);
  
  ws->def       = util_malloc(WELSPEC_NUM_KW * sizeof *ws->def,__func__);

  ws->name      = NULL;
  ws->group     = NULL;
  ws->fs_kw1    = NULL;
  ws->fs_kw2    = NULL;
  ws->ecl300_kw = NULL;

  return ws;
}



static void welspec_fwrite(const welspec_type * ws, FILE * stream)
{
  util_fwrite_string(ws->name     , stream);
  util_fwrite_string(ws->group    , stream);
  util_fwrite_string(ws->fs_kw1   , stream);
  util_fwrite_string(ws->fs_kw2   , stream);
  util_fwrite_string(ws->ecl300_kw, stream);

  util_fwrite(&ws->hh_i        , sizeof ws->hh_i        , 1 , stream, __func__);
  util_fwrite(&ws->hh_j        , sizeof ws->hh_j        , 1 , stream, __func__);
  util_fwrite(&ws->md          , sizeof ws->md          , 1 , stream, __func__);
  util_fwrite(&ws->phase       , sizeof ws->phase       , 1 , stream, __func__);
  util_fwrite(&ws->drain_rad   , sizeof ws->drain_rad   , 1 , stream, __func__);
  util_fwrite(&ws->inflow_eq   , sizeof ws->inflow_eq   , 1 , stream, __func__);
  util_fwrite(&ws->auto_shut   , sizeof ws->auto_shut   , 1 , stream, __func__);
  util_fwrite(&ws->crossflow   , sizeof ws->crossflow   , 1 , stream, __func__);
  util_fwrite(&ws->pvt_region  , sizeof ws->pvt_region  , 1 , stream, __func__);
  util_fwrite(&ws->hdstat_head , sizeof ws->hdstat_head , 1 , stream, __func__);
  util_fwrite(&ws->fip_region  , sizeof ws->fip_region  , 1 , stream, __func__);

  util_fwrite(ws->def          , sizeof ws->def         , WELSPEC_NUM_KW, stream, __func__);
}; 



static welspec_type * welspec_fread_alloc(FILE * stream)
{
  welspec_type * ws = welspec_alloc_empty();  

  ws->name      = util_fread_alloc_string(stream);
  ws->group     = util_fread_alloc_string(stream);
  ws->fs_kw1    = util_fread_alloc_string(stream);
  ws->fs_kw2    = util_fread_alloc_string(stream);
  ws->ecl300_kw = util_fread_alloc_string(stream);

  util_fread(&ws->hh_i        , sizeof ws->hh_i        , 1 , stream, __func__);
  util_fread(&ws->hh_j        , sizeof ws->hh_j        , 1 , stream, __func__);
  util_fread(&ws->md          , sizeof ws->md          , 1 , stream, __func__);
  util_fread(&ws->phase       , sizeof ws->phase       , 1 , stream, __func__);
  util_fread(&ws->drain_rad   , sizeof ws->drain_rad   , 1 , stream, __func__);
  util_fread(&ws->inflow_eq   , sizeof ws->inflow_eq   , 1 , stream, __func__);
  util_fread(&ws->auto_shut   , sizeof ws->auto_shut   , 1 , stream, __func__);
  util_fread(&ws->crossflow   , sizeof ws->crossflow   , 1 , stream, __func__);
  util_fread(&ws->pvt_region  , sizeof ws->pvt_region  , 1 , stream, __func__);
  util_fread(&ws->hdstat_head , sizeof ws->hdstat_head , 1 , stream, __func__);
  util_fread(&ws->fip_region  , sizeof ws->fip_region  , 1 , stream, __func__);

  util_fread(ws->def          , sizeof ws->def         , WELSPEC_NUM_KW, stream, __func__);

  return ws;
};



static void welspec_free(welspec_type * ws)
{
  if(ws->def != NULL) free(ws->def);

  if(ws->group      != NULL) free(ws->def      );
  if(ws->fs_kw1     != NULL) free(ws->fs_kw1   );
  if(ws->fs_kw2     != NULL) free(ws->fs_kw2   );
  if(ws->ecl300_kw  != NULL) free(ws->ecl300_kw);
};



static welspec_type * welspec_alloc_from_string(char ** token_list)
{
  welspec_type * ws = welspec_alloc_empty();

  {
    int i;
    for(i=0; i<WELSPEC_NUM_KW; i++)
    {
      if(token_list[i] == NULL)
        ws->def[i] = true;
      else
        ws->def[i] = false;
    }
  }

  ws->name = token_list[0];
  
  if(!ws->def[1])
    ws->group = util_alloc_string_copy(token_list[1]);

  ws->hh_i = sched_util_atoi(token_list[2]);
  ws->hh_j = sched_util_atoi(token_list[3]);

  if(!ws->def[4])
    ws->md = sched_util_atof(token_list[4]);

  if(!ws->def[5])
  {
    if(strcmp(token_list[5], PH_OIL_STRING) == 0)
      ws->phase = PH_OIL;
    else if(strcmp(token_list[5], PH_WAT_STRING) == 0)
      ws->phase = PH_WAT;
    else if(strcmp(token_list[5], PH_GAS_STRING) == 0)
      ws->phase = PH_GAS;
    else if(strcmp(token_list[5], PH_LIQ_STRING) == 0)
      ws->phase = PH_LIQ;
    else
      util_abort("%s: error when parsing WELSPECS. Phase %s not recognized - aborting.\n",__func__,token_list[5]);
  };

  if(!ws->def[6])
    ws->drain_rad = sched_util_atof(token_list[6]);

  if(!ws->def[7])
  {
    if(strcmp(token_list[7],IE_STD_STRING) == 0)
      ws->inflow_eq = IE_STD;
    else if(strcmp(token_list[7],IE_NO_STRING) == 0)
      ws->inflow_eq = IE_NO;
    else if(strcmp(token_list[7],IE_RG_STRING) == 0)
      ws->inflow_eq = IE_RG;
    else if(strcmp(token_list[7],IE_YES_STRING) == 0)
      ws->inflow_eq = IE_YES;
    else if(strcmp(token_list[7],IE_PP_STRING) == 0)
      ws->inflow_eq = IE_PP;
    else if(strcmp(token_list[7],IE_GPP_STRING) == 0)
      ws->inflow_eq = IE_GPP;
    else
      util_abort("%s: error when parsing WELSPECS. Inflow equation %s not recognized - aborting.\n",__func__,token_list[7]);
  }

  if(!ws->def[8])
  {
    if(strcmp(token_list[8],"STOP") == 0)
      ws->auto_shut = AS_STOP;
    else if(strcmp(token_list[8],"SHUT") == 0)
      ws->auto_shut = AS_SHUT;
    else
      util_abort("%s: error when parsing WELSPECS. Automatic shut-in mode %s not recognized - aborting.\n",__func__,token_list[8]);
  }

  if(!ws->def[9])
  {
    if(strcmp(token_list[9],"YES") == 0)
      ws->crossflow = CF_YES;
    else if(strcmp(token_list[9],"NO") == 0)
      ws->crossflow = CF_NO;
    else
      util_abort("%s: error when parsing WELSPECS. Crossflow ability mode %s not recognized - aborting.\n",__func__,token_list[9]);
  }

  if(!ws->def[10])
    ws->pvt_region = sched_util_atoi(token_list[10]);

  if(!ws->def[11])
  {
    if(strcmp(token_list[11],"SEG") == 0)
      ws->hdstat_head  = HD_SEG;
    else if(strcmp(token_list[11],"AVG") == 0)
      ws->hdstat_head  = HD_AVG;
    else
      util_abort("%s: error when parsing WELSPECS. Hydrostatic head model %s not recognized - aborting.\n",__func__,token_list[11]);
  }

  if(!ws->def[12])
    ws->fip_region = sched_util_atoi(token_list[12]);

  if(!ws->def[13])
    ws->fs_kw1 = util_alloc_string_copy(token_list[13]);

  if(!ws->def[14])
    ws->fs_kw2 = util_alloc_string_copy(token_list[14]);

  if(!ws->def[15])
    ws->ecl300_kw = util_alloc_string_copy(token_list[15]);

  return ws;
};
