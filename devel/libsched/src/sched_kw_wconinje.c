#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stringlist.h>
#include <util.h>
#include <vector.h>
#include <sched_kw_wconinje.h>
#include <sched_kw_untyped.h>
#include <sched_util.h>




/**
   
*/
   

typedef enum {OPEN  , STOP , SHUT , AUTO}      wconinje_status_enum;
#define STATUS_OPEN_STRING "OPEN"
#define STATUS_STOP_STRING "STOP"
#define STATUS_SHUT_STRING "SHUT"
#define STATUS_AUTO_STRING "AUTO"

  

typedef enum {WATER , GAS  , OIL}              wconinje_injector_enum;
#define TYPE_WATER_STRING "WATER"
#define TYPE_GAS_STRING   "GAS"
#define TYPE_OIL_STRING   "OIL"

  

typedef enum {RATE  , RESV , BHP , THP , GRUP} wconinje_control_enum;
#define CONTROL_RATE_STRING  "RATE"
#define CONTROL_RESV_STRING  "RESV"
#define CONTROL_BHP_STRING   "BHP"
#define CONTROL_THP_STRING   "THP"
#define CONTROL_GRUP_STRING  "GRUP"

#define SCHED_KW_WCONINJE_ID  99165
#define WCONINJE_NUM_KW 10
#define ECL_DEFAULT_KW "*"


struct sched_kw_wconinje_struct {
  int           __type_id;
  vector_type * wells;  /* A vector of wconinje_well_type instances. */
};



typedef struct {
  bool                       def[WCONINJE_NUM_KW];            /* Has the item been defaulted? */

  char                      * name;               /* This does NOT support well_name_root or well list notation. */
  wconinje_injector_enum      injector_type;      /* njecting GAS/WATER/OIL */
  wconinje_status_enum        status;             /* Well is open/shut/??? */
  wconinje_control_enum       control;            /* How is the well controlled? */
  double                      surface_flow;       
  double                      reservoir_flow;
  double                      BHP_target;
  double                      THP_target;
  int                         vfp_table_nr;
  double                      vapoil_conc;
} wconinje_well_type;
  



/*****************************************************************/
/* Implemeentation of the internal wconinje_well_type data type. */



static char * get_status_string(wconinje_status_enum status) 
{
  switch(status) {
  case(OPEN):
    return STATUS_OPEN_STRING; 
  case(STOP):
    return STATUS_STOP_STRING;
  case(SHUT):
    return STATUS_SHUT_STRING;
  case(AUTO):
    return STATUS_AUTO_STRING;
  default:
    return ECL_DEFAULT_KW;
  }
}


static char * get_injector_string(wconinje_injector_enum injector_type) {
  switch (injector_type) {
  case(WATER):
    return TYPE_WATER_STRING;
  case(GAS):
    return TYPE_GAS_STRING;
  case(OIL):
    return TYPE_OIL_STRING;
  default:
    return ECL_DEFAULT_KW;
  }
}


static char * get_control_string(wconinje_control_enum cmode)
{
  switch(cmode) {
  case(RATE):
    return CONTROL_RATE_STRING;
  case(RESV):
    return CONTROL_RESV_STRING;
  case(BHP):
    return CONTROL_BHP_STRING;
  case(THP):
    return CONTROL_THP_STRING;
  case(GRUP):
    return CONTROL_GRUP_STRING;
  default:
    return ECL_DEFAULT_KW;
  }
}



static wconinje_status_enum get_status_from_string(const char * st_string)
{
  if( strcmp(st_string, STATUS_OPEN_STRING) == 0)
    return OPEN; 
  else if( strcmp(st_string, STATUS_STOP_STRING) == 0)
    return STOP; 
  else if( strcmp(st_string, STATUS_SHUT_STRING) == 0)
    return SHUT; 
  else if( strcmp(st_string, STATUS_AUTO_STRING) == 0)
    return SHUT; 
  else
  {
    util_abort("%s: Could not recognize %s as a well status.\n", __func__, st_string);
    return 0;
  }
}


static wconinje_injector_enum get_type_from_string(const char * type_string) {
  if (strcmp(type_string , TYPE_WATER_STRING) == 0)
    return WATER;
  else if (strcmp(type_string , TYPE_GAS_STRING) == 0)
    return GAS;
  else if (strcmp(type_string , TYPE_OIL_STRING) == 0)
    return OIL;
  else {
    util_abort("%s: Could not recognize:%s as injector phase \n",__func__ , type_string);
    return 0;
  }
}



static wconinje_control_enum get_cmode_from_string(const char * cm_string)
{
  if(     strcmp(cm_string, CONTROL_RATE_STRING) == 0)
    return RATE;
  else if(strcmp(cm_string, CONTROL_RESV_STRING) == 0)
    return RESV;
  else if(strcmp(cm_string, CONTROL_BHP_STRING) == 0)
    return BHP;
  else if(strcmp(cm_string, CONTROL_THP_STRING) == 0)
    return THP;
  else if(strcmp(cm_string, CONTROL_GRUP_STRING) == 0)
    return GRUP;
  else
  {
    util_abort("%s: Could not recognize %s as a control mode.\n", __func__, cm_string);
    return 0;
  }
}




static wconinje_well_type * wconinje_well_alloc_empty()
{
  wconinje_well_type * well = util_malloc(sizeof * well, __func__);
  well->name = NULL;
  return well;
}



static void wconinje_well_free(wconinje_well_type * well)
{
  free(well->name);
  free(well);
}



static void wconinje_well_free__(void * well)
{
  wconinje_well_free( (wconinje_well_type *) well);
}


static wconinje_well_type * wconinje_well_alloc_from_string(char ** token_list)
{
  wconinje_well_type * well = wconinje_well_alloc_empty();

  {
    for(int i=0; i< WCONINJE_NUM_KW; i++)
    {
      if(token_list[i] == NULL)
        well->def[i] = true;
      else
        well->def[i] = false;
    }
  }

  well->name  = util_alloc_string_copy(token_list[0]);

  if(!well->def[1])
    well->injector_type = get_type_from_string(token_list[1]);

  if(!well->def[2])
    well->status = get_status_from_string(token_list[2]);

  if(!well->def[3])
    well->control = get_cmode_from_string(token_list[3]); 

  if(!well->def[4])
    well->surface_flow = sched_util_atof(token_list[4]); 

  if(!well->def[5])
    well->reservoir_flow = sched_util_atof(token_list[5]); 

  if(!well->def[6])
    well->BHP_target = sched_util_atof(token_list[7]);

  if(!well->def[7])
    well->THP_target = sched_util_atof(token_list[8]);
  
  if(!well->def[8])
    well->vfp_table_nr = sched_util_atoi(token_list[9]);
  
  if(!well->def[9])
    well->vapoil_conc = sched_util_atof(token_list[10]);

  return well;
}



static void wconinje_well_fprintf(const wconinje_well_type * well, FILE * stream)
{
  fprintf(stream, "  ");
  sched_util_fprintf_qst(well->def[0],  well->name                               , 8,  stream);
  sched_util_fprintf_qst(well->def[1],  get_injector_string(well->injector_type) , 5,  stream); /* 5 ?? */
  sched_util_fprintf_qst(well->def[2],  get_status_string(well->status)          , 4,  stream);
  sched_util_fprintf_qst(well->def[3],  get_control_string(well->control)        , 4,  stream);
  sched_util_fprintf_dbl(well->def[4],  well->surface_flow         , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[5],  well->reservoir_flow       , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[6],  well->BHP_target           , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[7],  well->THP_target           , 11, 3, stream);
  sched_util_fprintf_int(well->def[8],  well->vfp_table_nr         , 4,     stream);
  sched_util_fprintf_dbl(well->def[9],  well->vapoil_conc          , 11, 3, stream);
  fprintf(stream, "/ \n");
}


/*****************************************************************/




static sched_kw_wconinje_type * sched_kw_wconinje_alloc(bool alloc_untyped)
{
  sched_kw_wconinje_type * kw = util_malloc(sizeof * kw, __func__);
  kw->wells     = vector_alloc_new();
  kw->__type_id = SCHED_KW_WCONINJE_ID;
  return kw;
}

sched_kw_wconinje_type * sched_kw_wconinje_safe_cast( void * arg ) {
  sched_kw_wconinje_type * kw = (sched_kw_wconinje_type * ) arg;
  if (kw->__type_id == SCHED_KW_WCONINJE_ID)
    return kw;
  else {
    util_abort("%s: runtime cast failed \n",__func__);
    return NULL;
  }
}




void sched_kw_wconinje_free(sched_kw_wconinje_type * kw)
{
  vector_free( kw->wells );
  free(kw);
}



static void sched_kw_wconinje_add_line(sched_kw_wconinje_type * kw , const char * line , FILE * stream) {
  int tokens;
  char ** token_list;
  wconinje_well_type * well;

  sched_util_parse_line(line , &tokens , &token_list , WCONINJE_NUM_KW , NULL);
  well = wconinje_well_alloc_from_string( token_list );
  vector_append_owned_ref(kw->wells , well , wconinje_well_free__);
  util_free_stringlist( token_list , tokens );
}


sched_kw_wconinje_type * sched_kw_wconinje_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name)
{
  bool   at_eokw = false;
  char * line;
  sched_kw_wconinje_type * kw = sched_kw_wconinje_alloc(true);

  while(!*at_eof && !at_eokw)
  {
    line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);
    if(at_eokw)
    {
      break;
    }
    else if(*at_eof)
    {
      util_abort("%s: Reached EOF before WCONINJE was finished - aborting.\n", __func__);
    }
    else
    {
      sched_kw_wconinje_add_line(kw, line , stream);
      free(line);
    }
  }
  return kw;
}


void sched_kw_wconinje_fwrite(const sched_kw_wconinje_type *kw , FILE *stream) {
  util_abort("%s: not implemented \n",__func__);
}


sched_kw_wconinje_type *  sched_kw_wconinje_fread_alloc(FILE *stream) {
  util_abort("%s: not implemented \n",__func__);
  return NULL;
}


void sched_kw_wconinje_fprintf(const sched_kw_wconinje_type * kw , FILE * stream) {
  int size = vector_get_size(kw->wells);

  fprintf(stream, "WCONINJE\n");
  for(int i=0; i<size; i++)
  {
    const wconinje_well_type * well = vector_iget_const(kw->wells, i);
    wconinje_well_fprintf(well, stream);
  }
  fprintf(stream,"/\n\n");
}



char ** sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * kw , int * num_wells) {
  int size = vector_get_size(kw->wells);
  
  char ** well_names = util_malloc( size * sizeof * well_names , __func__);
  for(int i=0; i<size; i++)
  {
    const wconinje_well_type * well = vector_iget_const(kw->wells, i);
    well_names[i] = util_alloc_string_copy(well->name);
  }
  *num_wells = size;
  return well_names;
}




/*****************************************************************/
/* Functions exporting content to be used with the sched_file_update
   api.  */

/** Will return NULL if the well is not present. */
static wconinje_well_type * sched_kw_wconinje_get_well( const sched_kw_wconinje_type * kw , const char * well_name) {
  int size = vector_get_size(kw->wells);
  wconinje_well_type * well = NULL;
  int index = 0;
  do {
    wconinje_well_type * iwell = vector_iget( kw->wells , index);
    if (strcmp( well_name , iwell->name ) == 0) 
      well = iwell;
    
    index++;
  } while ((well == NULL) && (index < size));
  return well;
}



double sched_kw_wconinje_get_surface_flow( const sched_kw_wconinje_type * kw , const char * well_name) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL)
    return well->surface_flow;
  else
    return -1;
}

void sched_kw_wconinje_scale_surface_flow( const sched_kw_wconinje_type * kw , const char * well_name, double factor) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL)
    well->surface_flow *= factor;
}

void sched_kw_wconinje_set_surface_flow( const sched_kw_wconinje_type * kw , const char * well_name , double surface_flow) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL)
    well->surface_flow = surface_flow;
}


bool sched_kw_wconinje_has_well( const sched_kw_wconinje_type * kw , const char * well_name) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well == NULL)
    return false;
  else
    return true;
}


/*****************************************************************/

KW_FSCANF_ALLOC_IMPL(wconinje)
KW_FWRITE_IMPL(wconinje)
KW_FREAD_ALLOC_IMPL(wconinje)
KW_FREE_IMPL(wconinje)
KW_FPRINTF_IMPL(wconinje)
