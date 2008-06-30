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

typedef enum {OIL,  WATER, GAS, LIQ}           phase_type;
typedef enum {STD,  NO   , RG, YES, PP, GPP}   inflow_eq_type;
typedef enum {STOP, SHUT}                      auto_shut_type;
typedef enum {CF_YES,  CF_NO}                  crossflow_type; 
typedef enum {SEG,  AVG}                       hdstat_head_type;

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
    if(strcmp(token_list[5], "OIL") == 0)
      ws->phase = OIL;
    else if(strcmp(token_list[5], "WATER") == 0)
      ws->phase = WATER;
    else if(strcmp(token_list[5], "GAS") == 0)
      ws->phase = GAS;
    else if(strcmp(token_list[5], "LIQ") == 0)
      ws->phase = LIQ;
    else
      util_abort("%s: error when parsing WELSPECS. Phase %s not recognized - aborting.\n",__func__,token_list[5]);
  };

  if(!ws->def[6])
    ws->drain_rad = sched_util_atof(token_list[6]);

  if(!ws->def[7])
  {
    if(strcmp(token_list[7],"STD") == 0)
      ws->inflow_eq = STD;
    else if(strcmp(token_list[7],"NO") == 0)
      ws->inflow_eq = NO;
    else if(strcmp(token_list[7],"R-G") == 0)
      ws->inflow_eq = RG;
    else if(strcmp(token_list[7],"YES") == 0)
      ws->inflow_eq = YES;
    else if(strcmp(token_list[7],"P-P") == 0)
      ws->inflow_eq = PP;
    else if(strcmp(token_list[7],"GPP") == 0)
      ws->inflow_eq = GPP;
    else
      util_abort("%s: error when parsing WELSPECS. Inflow equation %s not recognized - aborting.\n",__func__,token_list[7]);
  }

  if(!ws->def[8])
  {
    if(strcmp(token_list[8],"STOP") == 0)
      ws->auto_shut = STOP;
    else if(strcmp(token_list[8],"SHUT") == 0)
      ws->auto_shut = SHUT;
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
      ws->hdstat_head  = SEG;
    else if(strcmp(token_list[11],"AVG") == 0)
      ws->hdstat_head  = AVG;
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
