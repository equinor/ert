#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stringlist.h>
#include <util.h>
#include <vector.h>
#include <sched_kw_wconinje.h>
#include <sched_util.h>
#include <stringlist.h>
#include <sched_types.h>

#define DEFAULT_INJECTOR_STATE OPEN

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
  sched_phase_type            injector_type;      /* Injecting GAS/WATER/OIL */
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
  if (strcmp( st_string , SCHED_KW_DEFAULT_ITEM ) == 0)
    return DEFAULT_INJECTOR_STATE;
  else if( strcmp(st_string, STATUS_OPEN_STRING) == 0)
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
    util_abort("%s: Could not recognize %s as a control mode. Valid values are: [%s, %s, %s, %s, %s] \n", __func__, cm_string, 
	       CONTROL_RATE_STRING , CONTROL_RESV_STRING , CONTROL_BHP_STRING, CONTROL_THP_STRING, CONTROL_GRUP_STRING);
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





static wconinje_well_type * wconinje_well_alloc_from_tokens(const stringlist_type * line_tokens ) {
  wconinje_well_type * well = wconinje_well_alloc_empty();
  sched_util_init_default( line_tokens , well->def );

  well->name           = util_alloc_string_copy( stringlist_iget( line_tokens , 0 ));
  well->injector_type  = sched_phase_type_from_string(stringlist_iget(line_tokens , 1));
  well->status         = get_status_from_string( stringlist_iget( line_tokens , 2 ));
  well->control        = get_cmode_from_string( stringlist_iget( line_tokens , 3 ));
  well->surface_flow   = sched_util_atof( stringlist_iget( line_tokens , 4 ));
  well->reservoir_flow = sched_util_atof(stringlist_iget(line_tokens , 5 ));
  well->BHP_target     = sched_util_atof(stringlist_iget(line_tokens , 6 ));
  well->THP_target     = sched_util_atof( stringlist_iget( line_tokens , 7 ));
  well->vfp_table_nr   = sched_util_atoi( stringlist_iget( line_tokens , 8));
  well->vapoil_conc    = sched_util_atof( stringlist_iget( line_tokens , 9 ));

  return well;
}



static void wconinje_well_fprintf(const wconinje_well_type * well, FILE * stream)
{
  fprintf(stream, "  ");
  sched_util_fprintf_qst(well->def[0],  well->name                                   , 8,     stream);
  sched_util_fprintf_qst(well->def[1],  sched_phase_type_string(well->injector_type) , 5,     stream); /* 5 ?? */
  sched_util_fprintf_qst(well->def[2],  get_status_string(well->status)              , 4,     stream);
  sched_util_fprintf_qst(well->def[3],  get_control_string(well->control)            , 4,     stream);
  sched_util_fprintf_dbl(well->def[4],  well->surface_flow                           , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[5],  well->reservoir_flow                         , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[6],  well->BHP_target                             , 11, 3, stream);
  sched_util_fprintf_dbl(well->def[7],  well->THP_target                             , 11, 3, stream);
  sched_util_fprintf_int(well->def[8],  well->vfp_table_nr                           , 4,     stream);
  sched_util_fprintf_dbl(well->def[9],  well->vapoil_conc                            , 11, 3, stream);
  fprintf(stream, "/ \n");
}


/*****************************************************************/




static sched_kw_wconinje_type * sched_kw_wconinje_alloc_empty() {
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


static void sched_kw_wconinje_add_well( sched_kw_wconinje_type * kw , const wconinje_well_type * well) {
  vector_append_owned_ref(kw->wells , well , wconinje_well_free__);  
}





sched_kw_wconinje_type * sched_kw_wconinje_alloc(const stringlist_type * tokens , int * token_index ) {
  sched_kw_wconinje_type * kw = sched_kw_wconinje_alloc_empty();
  int eokw                    = false;
  do {
    stringlist_type * line_tokens = sched_util_alloc_line_tokens( tokens , false , WCONINJE_NUM_KW , token_index );
    if (line_tokens == NULL)
      eokw = true;
    else {
      wconinje_well_type * well = wconinje_well_alloc_from_tokens( line_tokens );
      sched_kw_wconinje_add_well( kw , well );
      stringlist_free( line_tokens );
    } 
  } while (!eokw);
  return kw;  
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



void sched_kw_wconinje_shift_surface_flow( const sched_kw_wconinje_type * kw , const char * well_name , double delta_surface_flow) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL)
    well->surface_flow += delta_surface_flow;
}


sched_phase_type sched_kw_wconinje_get_phase( const sched_kw_wconinje_type * kw , const char * well_name) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL)
    return well->injector_type;
  else
    return -1;
}



bool sched_kw_wconinje_has_well( const sched_kw_wconinje_type * kw , const char * well_name) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well == NULL)
    return false;
  else
    return true;
}


sched_kw_wconinje_type * sched_kw_wconinje_copyc(const sched_kw_wconinje_type * kw) {
  util_abort("%s: not implemented ... \n",__func__);
  return NULL;
}


bool sched_kw_wconinje_well_open( const sched_kw_wconinje_type * kw, const char * well_name) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well == NULL)
    return false;
  else {
    /* OK - we have the well. */

    if (well->status == OPEN) {
      /* The well seems to be open - any rates around? */
      if (well->surface_flow > 0)
        return true;
      else
        return false;
    } else
      return false;
  }
}


/*****************************************************************/

KW_IMPL(wconinje)
