/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sched_kw_wconinje.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <ert/util/stringlist.hpp>
#include <ert/util/util.hpp>
#include <ert/util/vector.hpp>
#include <ert/util/buffer.hpp>
#include <ert/util/int_vector.hpp>
#include <ert/util/double_vector.hpp>

#include <ert/sched/sched_kw_wconinje.hpp>
#include <ert/sched/sched_util.hpp>
#include <ert/sched/sched_types.hpp>


#define DEFAULT_INJECTOR_STATE OPEN

#define SCHED_KW_WCONINJE_ID  99165
#define WCONINJE_TYPE_ID      5705235
#define WCONINJE_NUM_KW 10
#define ECL_DEFAULT_KW "*"


struct sched_kw_wconinje_struct {
  int           __type_id;
  vector_type * wells;  /* A vector of wconinje_well_type instances. */
};



typedef struct {
  bool                       def[WCONINJE_NUM_KW];            /* Has the item been defaulted? */

  char                      * name;               /* This does NOT support well_name_root or well list notation. */
  sched_phase_enum            injector_type;      /* Injecting GAS/WATER/OIL */
  well_status_enum            status;             /* Well is open/shut/??? */
  well_cm_enum                cmode;              /* How is the well controlled? */
  double                      surface_flow;
  double                      reservoir_flow;
  double                      BHP_target;
  double                      THP_target;
  int                         vfp_table_nr;
  double                      vapoil_conc;
} wconinje_well_type;





struct wconinje_state_struct {
  UTIL_TYPE_ID_DECLARATION;
  char *               well_name;
  const time_t_vector_type * time;
  int_vector_type    * phase;                  /* Contains values from sched_phase_enum */
  int_vector_type    * state;                  /* Contains values from the well_status_enum. */
  int_vector_type    * cmode;                  /* Contains values from the well_cm_enum. */
  double_vector_type * surface_flow;
  double_vector_type * reservoir_flow;
  double_vector_type * bhp_limit;
  double_vector_type * thp_limit;
  int_vector_type    * vfp_table_nr;
  double_vector_type * vapoil;
};




/*****************************************************************/
/* Implemeentation of the internal wconinje_well_type data type. */










static wconinje_well_type * wconinje_well_alloc_empty()
{
  wconinje_well_type * well = (wconinje_well_type*)util_malloc(sizeof * well);
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
  well->cmode          = sched_types_get_cm_from_string( stringlist_iget( line_tokens , 3 ) , false);
  well->surface_flow   = sched_util_atof( stringlist_iget( line_tokens , 4 ));
  well->reservoir_flow = sched_util_atof(stringlist_iget(line_tokens , 5 ));
  well->BHP_target     = sched_util_atof(stringlist_iget(line_tokens , 6 ));
  well->THP_target     = sched_util_atof( stringlist_iget( line_tokens , 7 ));
  well->vfp_table_nr   = sched_util_atoi( stringlist_iget( line_tokens , 8));
  well->vapoil_conc    = sched_util_atof( stringlist_iget( line_tokens , 9 ));

  well->status         = sched_types_get_status_from_string( stringlist_iget( line_tokens , 2 ));
  if (well->status == DEFAULT)
    well->status = DEFAULT_INJECTOR_STATE;
  return well;
}



static void wconinje_well_fprintf(const wconinje_well_type * well, FILE * stream)
{
  fprintf(stream, "  ");
  sched_util_fprintf_qst(well->def[0],  well->name                                   , 8,     stream);
  sched_util_fprintf_qst(well->def[1],  sched_phase_type_string(well->injector_type) , 5,     stream); /* 5 ?? */
  sched_util_fprintf_qst(well->def[2],  sched_types_get_status_string(well->status)  , 4,     stream);
  sched_util_fprintf_qst(well->def[3],  sched_types_get_cm_string(well->cmode)       , 4,     stream);
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
  sched_kw_wconinje_type * kw = (sched_kw_wconinje_type*)util_malloc(sizeof * kw);
  kw->wells     = vector_alloc_new();
  kw->__type_id = SCHED_KW_WCONINJE_ID;
  return kw;
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
    const wconinje_well_type * well = (const wconinje_well_type*)vector_iget_const(kw->wells, i);
    wconinje_well_fprintf(well, stream);
  }
  fprintf(stream,"/\n\n");
}



char ** sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * kw , int * num_wells) {
  int size = vector_get_size(kw->wells);

  char ** well_names = (char**)util_malloc( size * sizeof * well_names );
  for(int i=0; i<size; i++)
  {
    const wconinje_well_type * well = (const wconinje_well_type*)vector_iget_const(kw->wells, i);
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
    wconinje_well_type * iwell = (wconinje_well_type*)vector_iget( kw->wells , index);
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
      return false;  }
}

/*****************************************************************/


/*****************************************************************/

wconinje_state_type * wconinje_state_alloc( const char * well_name , const time_t_vector_type * time) {
  wconinje_state_type * wconinje = (wconinje_state_type*)util_malloc( sizeof * wconinje);
  UTIL_TYPE_ID_INIT( wconinje , WCONINJE_TYPE_ID );

  wconinje->phase          = int_vector_alloc( 0 , 0 );
  wconinje->state          = int_vector_alloc( 0 , 0 );  /* Default wconinje state ? */
  wconinje->cmode          = int_vector_alloc( 0 , 0 );  /* Default control mode ?? */
  wconinje->surface_flow   = double_vector_alloc( 0 , 0 );
  wconinje->reservoir_flow = double_vector_alloc( 0 , 0 );
  wconinje->bhp_limit      = double_vector_alloc( 0 , 0 );
  wconinje->thp_limit      = double_vector_alloc( 0 , 0 );
  wconinje->vfp_table_nr   = int_vector_alloc( 0 , 0 );
  wconinje->vapoil         = double_vector_alloc( 0 ,0 );
  wconinje->time           = time;
  wconinje->well_name      = util_alloc_string_copy( well_name );

  return wconinje;
}


static UTIL_SAFE_CAST_FUNCTION( wconinje_state , WCONINJE_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( wconinje_state , WCONINJE_TYPE_ID )

void wconinje_state_free( wconinje_state_type * wconinje ) {

  int_vector_free(wconinje->phase);
  int_vector_free(wconinje->state);
  int_vector_free(wconinje->cmode);
  double_vector_free(wconinje->surface_flow);
  double_vector_free(wconinje->reservoir_flow);
  double_vector_free(wconinje->bhp_limit);
  double_vector_free(wconinje->thp_limit);
  int_vector_free(wconinje->vfp_table_nr);
  double_vector_free(wconinje->vapoil);
  free( wconinje->well_name );
  free( wconinje );

}


void sched_kw_wconinje_update_state( const sched_kw_wconinje_type * kw , wconinje_state_type * state , const char * well_name , int report_step ) {
  wconinje_well_type * well = sched_kw_wconinje_get_well( kw , well_name );
  if (well != NULL) {
    int_vector_iset_default(state->phase             , report_step , well->injector_type );
    int_vector_iset_default(state->state             , report_step , well->status);
    int_vector_iset_default(state->cmode             , report_step , well->cmode);
    double_vector_iset_default(state->surface_flow   , report_step , well->surface_flow);
    double_vector_iset_default(state->reservoir_flow , report_step , well->reservoir_flow);
    double_vector_iset_default(state->bhp_limit      , report_step , well->BHP_target);
    double_vector_iset_default(state->thp_limit      , report_step , well->THP_target);
    int_vector_iset_default(state->vfp_table_nr      , report_step , well->vfp_table_nr);
    double_vector_iset_default(state->vapoil         , report_step , well->vapoil_conc);
  }
}


void sched_kw_wconinje_close_state(wconinje_state_type * state , int report_step ) {
  fprintf(stderr,"** Warning: %s not implemented \n",__func__);
}





KW_IMPL(wconinje)
