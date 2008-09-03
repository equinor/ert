#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sched_kw_welspecs.h>
#include <list.h>
#include <util.h>
#include <sched_util.h>
#include <sched_macros.h>

/*
  Define the maximum number of keywords in a WELSPEC record.
  Note that this includes FrontSim and ECLIPSE 300 KWs.
*/
#define WELSPEC_NUM_KW 16
#define ECL_DEFAULT_KW "*"


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
    def : Read as defaulted, not as defined.
  */
  bool             def[WELSPEC_NUM_KW];

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



static char * get_phase_string(phase_type phase)
{
  switch(phase)
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
      return ECL_DEFAULT_KW;
  }
};



static char * get_inflow_eq_string(inflow_eq_type eq)
{
  switch(eq)
  {
    case(IE_STD):
      return IE_STD_STRING;
    case(IE_NO):
      return IE_NO_STRING;
    case(IE_RG):
      return IE_NO_STRING;
    case(IE_YES):
      return IE_YES_STRING;
    case(IE_PP):
      return IE_PP_STRING;
    case(IE_GPP):
      return IE_GPP_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
};



static char * get_auto_shut_string(auto_shut_type as)
{
  switch(as)
  {
    case(AS_STOP):
      return AS_STOP_STRING;
    case(AS_SHUT):
      return AS_SHUT_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
};



static char * get_crossflow_string(crossflow_type cf)
{
  switch(cf)
  {
    case(CF_YES):
      return CF_YES_STRING;
    case(CF_NO):
      return CF_NO_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
}



static char * get_hdstat_head_string(hdstat_head_type hd)
{
  switch(hd)
  {
    case(HD_SEG):
      return HD_SEG_STRING;
    case(HD_AVG):
      return HD_AVG_STRING;
    default:
      return ECL_DEFAULT_KW;
  }
};



static void welspec_sched_fprintf(const welspec_type * ws, FILE * stream)
{
  fprintf(stream, " ");
  sched_util_fprintf_qst(ws->def[0]  , ws->name                                , 8,     stream);
  sched_util_fprintf_qst(ws->def[1]  , ws->group                               , 8,     stream);
  sched_util_fprintf_int(ws->def[2]  , ws->hh_i                                , 4,     stream);
  sched_util_fprintf_int(ws->def[3]  , ws->hh_j                                , 4,     stream);
  sched_util_fprintf_dbl(ws->def[4]  , ws->md                                  , 8, 3,  stream);
  sched_util_fprintf_qst(ws->def[5]  , get_phase_string(ws->phase)             , 5,     stream);
  sched_util_fprintf_dbl(ws->def[6]  , ws->drain_rad                           , 8, 3,  stream);
  sched_util_fprintf_qst(ws->def[7]  , get_inflow_eq_string(ws->inflow_eq)     , 3,     stream);
  sched_util_fprintf_qst(ws->def[8]  , get_auto_shut_string(ws->auto_shut)     , 4,     stream);
  sched_util_fprintf_qst(ws->def[9]  , get_crossflow_string(ws->crossflow)     , 3,     stream);
  sched_util_fprintf_int(ws->def[10] , ws->pvt_region                          , 4,     stream);
  sched_util_fprintf_qst(ws->def[11] , get_hdstat_head_string(ws->hdstat_head) , 3,     stream);
  sched_util_fprintf_int(ws->def[12] , ws->fip_region                          , 4,     stream);
  /*
  sched_util_fprintf_qst(ws->def[13] , ws->fs_kw1                              , 8,     stream);
  sched_util_fprintf_qst(ws->def[14] , ws->fs_kw2                              , 8,     stream);
  sched_util_fprintf_qst(ws->def[15] , ws->ecl300_kw                           , 8,     stream);
  */
  fprintf(stream,"/\n");
};



static welspec_type * welspec_alloc_empty()
{
  welspec_type *ws = util_malloc(sizeof *ws,__func__);
  
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

  util_fwrite(&ws->def         , sizeof ws->def[0], WELSPEC_NUM_KW, stream, __func__);
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

  util_fread(&ws->def         , sizeof ws->def[0]         , WELSPEC_NUM_KW, stream, __func__);

  return ws;
};




static void welspec_free(welspec_type * ws)
{
  free(ws->group    );
  util_safe_free(ws->fs_kw1   );
  util_safe_free(ws->fs_kw2   );
  util_safe_free(ws->ecl300_kw);
  free(ws->name);
  free(ws);
};



static void welspec_free__(void * __ws)
{
  welspec_type * ws = (welspec_type *) __ws;
  welspec_free(ws);
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

  ws->name = util_alloc_string_copy(token_list[0]);
  
  if(!ws->def[1])
    ws->group = util_alloc_string_copy(token_list[1]);

  ws->hh_i = sched_util_atoi(token_list[2]);
  ws->hh_j = sched_util_atoi(token_list[3]);

  if(!ws->def[4])
    ws->md = sched_util_atof(token_list[4]);

  if(!ws->def[5])
  {
    if(strcmp(token_list[5]     , PH_OIL_STRING) == 0)
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
    if(strcmp(token_list[7]     ,IE_STD_STRING) == 0)
      ws->inflow_eq = IE_STD;
    else if(strcmp(token_list[7],IE_NO_STRING)  == 0)
      ws->inflow_eq = IE_NO;
    else if(strcmp(token_list[7],IE_RG_STRING)  == 0)
      ws->inflow_eq = IE_RG;
    else if(strcmp(token_list[7],IE_YES_STRING) == 0)
      ws->inflow_eq = IE_YES;
    else if(strcmp(token_list[7],IE_PP_STRING)  == 0)
      ws->inflow_eq = IE_PP;
    else if(strcmp(token_list[7],IE_GPP_STRING) == 0)
      ws->inflow_eq = IE_GPP;
    else
      util_abort("%s: error when parsing WELSPECS. Inflow equation %s not recognized - aborting.\n",__func__,token_list[7]);
  }

  if(!ws->def[8])
  {
    if(strcmp(token_list[8],     AS_STOP_STRING) == 0)
      ws->auto_shut = AS_STOP;
    else if(strcmp(token_list[8],AS_SHUT_STRING) == 0)
      ws->auto_shut = AS_SHUT;
    else
      util_abort("%s: error when parsing WELSPECS. Automatic shut-in mode %s not recognized - aborting.\n",__func__,token_list[8]);
  }

  if(!ws->def[9])
  {
    if(strcmp(token_list[9]     ,CF_YES_STRING) == 0)
      ws->crossflow = CF_YES;
    else if(strcmp(token_list[9],CF_NO_STRING) == 0)
      ws->crossflow = CF_NO;
    else
      util_abort("%s: error when parsing WELSPECS. Crossflow ability mode %s not recognized - aborting.\n",__func__,token_list[9]);
  }

  if(!ws->def[10])
    ws->pvt_region = sched_util_atoi(token_list[10]);

  if(!ws->def[11])
  {
    if(strcmp(token_list[11]     ,HD_SEG_STRING) == 0)
      ws->hdstat_head  = HD_SEG;
    else if(strcmp(token_list[11],HD_AVG_STRING) == 0)
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



static sched_kw_welspecs_type * sched_kw_welspecs_alloc()
{
  sched_kw_welspecs_type * kw = util_malloc(sizeof * kw,__func__);
  kw->welspec_list = list_alloc();
  return kw;
};



static void sched_kw_welspecs_add_line(sched_kw_welspecs_type * kw, const char * line)
{
  int tokens;
  char **token_list;

  sched_util_parse_line(line, &tokens, &token_list, WELSPEC_NUM_KW, NULL);
  {
    welspec_type * ws = welspec_alloc_from_string(token_list);
    list_append_list_owned_ref(kw->welspec_list, ws, welspec_free__);
  }
  sched_util_free_token_list(tokens,token_list);
};



/*****************************************************************************/


sched_kw_welspecs_type * sched_kw_welspecs_fscanf_alloc(FILE *stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_welspecs_type * kw = sched_kw_welspecs_alloc();

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before WELSPECS was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_welspecs_add_line(kw, line);
      free(line);
    }
  }

  return kw;

}



void sched_kw_welspecs_free(sched_kw_welspecs_type * kw)
{
  list_free(kw->welspec_list);
  free(kw);
};



void sched_kw_welspecs_fprintf(const sched_kw_welspecs_type * kw, FILE * stream)
{
  fprintf(stream, "WELSPECS\n");
  list_node_type *ws_node = list_get_head(kw->welspec_list);
  while(ws_node != NULL)
  {
   welspec_type * ws = list_node_value_ptr(ws_node);
   welspec_sched_fprintf(ws, stream);
   ws_node = list_node_get_next(ws_node);
  }
  fprintf(stream,"/\n\n");
};



void sched_kw_welspecs_fwrite(const sched_kw_welspecs_type * kw, FILE * stream)
{
  int welspec_lines = list_get_size(kw->welspec_list);
  util_fwrite(&welspec_lines, sizeof welspec_lines, 1, stream, __func__);
  {
    list_node_type * ws_node = list_get_head(kw->welspec_list);
    while(ws_node != NULL)
    {
      welspec_type * ws = list_node_value_ptr(ws_node);
      welspec_fwrite(ws, stream);
      ws_node = list_node_get_next(ws_node);
    }
  }
};



sched_kw_welspecs_type * sched_kw_welspecs_fread_alloc(FILE * stream)
{
  int i, welspec_lines;
  sched_kw_welspecs_type * kw = sched_kw_welspecs_alloc();
  
  util_fread(&welspec_lines, sizeof welspec_lines, 1, stream, __func__);

  for(i=0; i< welspec_lines; i++)
  {
    welspec_type * ws = welspec_fread_alloc(stream);
    list_append_list_owned_ref(kw->welspec_list, ws, welspec_free__);
  }

  return kw;
};



/***********************************************************************/

KW_FSCANF_ALLOC_IMPL(welspecs)
KW_FWRITE_IMPL(welspecs)
KW_FREAD_ALLOC_IMPL(welspecs)
KW_FREE_IMPL(welspecs)
KW_FPRINTF_IMPL(welspecs)
