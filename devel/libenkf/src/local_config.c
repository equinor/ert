#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector.h>
#include <util.h>
#include <ecl_grid.h>
#include <ecl_region.h>
#include <local_ministep.h>
#include <local_updatestep.h>
#include <local_config.h>
#include <int_vector.h>
#include <ensemble_config.h>
#include <enkf_obs.h>
#include "config_keys.h"
#include "enkf_defaults.h"
/******************************************************************/
/*

To create a configuration for localization you must "program" your own
configuration file, this file is then loaded from the ert/enkf proper
application. The 'commands' available in the local_config programming
language are listed below. [In the listing below the arguments to the
functions are shown in square braces, these braces should NOT be
included in the program file:


CREATE_UPDATESTEP [NAME_OF_UPDATESTEP]
--------------------------------------- 
This function will create a new updatestep with the name
'NAME_OF_UPDATESTEP'. Observe that you must add (at least) one
ministep to the updatestep, otherwise it will not be able to do
anything.


CREATE_MINISTEP [NAME_OF_MINISTEP]
----------------------------------
This function will create a new ministep with the name
'NAME_OF_MINISTEP'. The ministep is then ready for adding data and
observation keys. Before the ministep will be used you must attach it
to an updatestep with the ATTACH_MINISTEP command


ATTACH_MINISTEP [NAME_OF_UPDATESTEP  NAME_OF_MINISTEP]
------------------------------------------------------
This function will attach the ministep 'NAME_OF_MINISTEP' to the
updatestep 'NAME_OF_UPDATESTEP'; one ministep can be attached to many
updatesteps.


ADD_DATA [NAME_OF_MINISTEP   KEY]
---------------------------------
This function will install 'KEY' as one enkf node which should be
updated in this ministep. If you do not manipulate the KEY further
with the ACTIVE_LIST_ADD_DATA_INDEX function the KEY will be added as
'ALL_ACTIVE', i.e. all elements will be updated.


ADD_OBS [NAME_OF_MINISTEP  OBS_KEY]
-----------------------------------
This function will install the observation 'OBS_KEY' as an observation
for this ministep - similarly to the ADD_DATA function.


DEL_DATA [NAME_OF_MINISTEP  KEY]
--------------------------------
This function will delete the data 'KEY' from the ministep
'NAME_OF_MINISTEP'.


DEL_OBS [NAME_OF_MINISTEP  OBS_KEY]
-----------------------------------
This function will delete the obs 'OBS_KEY' from the ministep
'NAME_OF_MINISTEP'.


DEL_ALL_DATA [NAME_OF_MINISTEP]
--------------------------------
This function will delete all the data keys from the ministep
'NAME_OF_MINISTEP'; typically used after a call to
alloc_ministep_copy.


DEL_ALL_OBS [NAME_OF_MINISTEP]
-----------------------------------
this function will delete all the obs keys from the ministep
'NAME_OF_MINISTEP'; typically used after a call to
alloc_ministep_copy.


ALLOC_MINISTEP_COPY [SRC_MINISTEP  TARGET_MINISTEP]
---------------------------------------------------
This function will create a new ministep 'TARGET_MINISTEP' which is a
copy of the original ministep 'SRC_MINISTEP'. You can then add/delete
nodes to the two ministep instances separately.


ACTIVE_LIST_ADD_OBS_INDEX[MINISTEP_NAME  OBS_KEY  INDEX]
--------------------------------------------------------
This function will say that the observation with name 'OBS_KEY' in
ministep with name 'MINISTEP_NAME' should have the index 'INDEX'
active.


ACTIVE_LIST_ADD_DATA_INDEX[MINISTEP_NAME  DATA_KEY  INDEX]
--------------------------------------------------------
This function will say that the data with name 'DATA_KEY' in ministep
with name 'MINISTEP_NAME' should have the index 'INDEX' active.


ACTIVE_LIST_ADD_MANY_OBS_INDEX[MINISTEP_NAME  OBS_KEY  N INDEX1 INDEX2 INDEX3 .. INDEXN]
----------------------------------------------------------------------------------------
This function is simular to ACTIVE_LIST_ADD_OBS_INDEX, but it will add many indices.



ACTIVE_LIST_ADD_MANY_DATA_INDEX[MINISTEP_NAME  DATA_KEY  N INDEX1 INDEX2 INDEX3 .. INDEXN]
------------------------------------------------------------------------------------------
This function is simular to ACTIVE_LIST_ADD_DATA_INDEX, but it will add many indices.


INSTALL_UPDATESTEP [NAME_OF_UPDATESTEP  STEP1   STEP2]
----------------------------------------------------
This function will install the updatestep 'NAME_OF_UPDATESTEP' for the
report steps [STEP1,..,STEP2].


INSTALL_DEFAULT_UPDATESTEP [NAME_OF_UPDATESTEP]
-----------------------------------------------
This function will install 'NAME_OF_UPDATESTEP' as the default
updatestep which applies to all report streps where you have not
explicitly set another updatestep with the INSTALL_UPDATESTEP function.



ADD_FIELD   [MINISTEP_NAME    FIELD_NAME    REGION_NAME]
--------------------------------------------------------

This function will install the node with name 'FIELD_NAME' in the
ministep 'MINISTEP_NAME'. It will in addition select all the
(currently) active cells in the region 'REGION_NAME' as active for
this field/ministep combination. The ADD_FIELD command is actually a
shortcut for the following:

   ADD_DATA   MINISTEP  FIELD_NAME
   ACTIVE_LIST_AD_MANY_DATA_INDEX  <All the indices from the region>



LOAD_FILE       [KEY    FILENAME]
---------------------------------
This function will load an ECLIPSE file in restart format
(i.e. restart file or INIT file), the keywords in this file can then
subsequently be used in REGION_SELECT_VALUE_XXX commands below. The
'KEY' argument is a string which will be used later when we refer to
the content of this file


CREATE_REGION   [REGION_NAME    SELECT_ALL]
-------------------------------------------
This function will create a new region 'REGION_NAME', which can
subsequently be used when defining active regions for fields. The
second argument, SELECT_ALL, is a boolean value. If this value is set
to true the region will start with all cells selected, if set to false
the region will start with no cells selected.


REGION_SELECT_ALL     [REGION_NAME   SELECT]
--------------------------------------------
Will select all the cells in the region (or deselect if SELECT == FALSE).


REGION_SELECT_VALUE_EQUAL   [REGION_NAME   FILE_KEY:KEYWORD<:NR>    VALUE   SELECT]
-----------------------------------------------------------------------------------
This function will compare an ecl_kw instance loaded from file with a
user supplied value, and select (or deselect) all cells which match
this value. It is assumed that the ECLIPSE keyword is an INTEGER
keyword, for float comparisons use the REGION_SELECT_VALUE_LESS and
REGION_SELECT_VALUE_MORE functions.


REGION_SELECT_VALUE_LESS
REGION_SELECT_VALUE_MORE    [REGION_NAME   FILE_KEY:KEYWORD<:NR>  VALUE   SELECT]
---------------------------------------------------------------------------------
This function will compare an ecl_kw instance loaded from disc with a
numerical value, and select all cells which have numerical below or
above the limiting value. The ecl_kw value should be a floating point
value like e.g. PRESSURE or PORO. The arguments are just as for REGION_SELECT_VALUE_EQUAL.


REGION_SELECT_BOX            [ REGION_NAME i1 i2 j1 j2 k1 k2 SELECT]
--------------------------------------------------------------------
This function will select (or deselect) all the cells in the box
defined by the six coordinates i1 i2 j1 j2 k1 k2. The coordinates are
inclusive, and the counting starts at 1.


REGION_SELECT_SLICE         [ REGION_NAME dir n1 n2 SELECT]
-----------------------------------------------------------
This ffunction will select a slice in the direction given by 'dir',
which can 'x', 'y' or 'z'. Depending on the value of 'dir' the numbers
n1 and n2 are interpreted as (i1 i2), (j1 j2) or (k1 k2)
respectively. The numbers n1 and n2 are inclusice and the counting
starts at 1. It is OK to use very high/low values to imply "the rest
of the cells" in one direction.




I have added comments in the example - that is not actually supported (yet at least)

-------------------------------------------------------------------------------------
CREATE_MINISTEP MSTEP
CREATE_REGION   FIPNUM3       FALSE              --- We create a region called FIPNUM3 with no elements 
                                                 --- selected from the start.
CREATE_REGION   WATER_FLOODED TRUE               --- We create a region called WATER_FLOEDED, 
                                                 --- which starts with all elements selected.
CREATE_REGION   MIDLLE        FALSE              --- Create a region called MIDDLE with
                                                 --- no elements initially.    
LOAD_FILE       INIT          /path/to/ECL.INIT  --- We load the INIT file and label
                                                 --- it as INIT for further use.
LOAD_FILE       RESTART       /path/to/ECL.UNRST --- We load a unified restart fila
                                                 --- and label it RESTART

-- We select all the cells corresponding to a FIPNUM value of 3. Since there is
-- only one FIPNUM keyword in the INIT file we do not need the :NR suffix on the key.
REGION_SELECT_VALUE_EQUAL     FIPNUM3     INIT:FIPNUM    3    TRUE


-- In the region WATER_FLOODED all cells are selected from the start, now
-- we deselect all the cells which have SWAT value below 0.90, at report step 100:
REGION_SELECT_VALUE_LESS    WATER_FLOODED RESTART:SWAT:100   0.90    FALSE

-- We select the the layers k=4,5,6 in the region MIDDLE. The indices 4,5
-- and 6 are "normal" k values, where the counting starts at 1.
REGION_SELECT_SLICE  MIDDLE   Z   4  6   TRUE   


-- We add field data in the current ministep, corresponding to the two 
-- selection region (poro is only updated in FIPNUM3, PERMX is only updated in 
-- the water flooded region and NTG is only updates in the MIDDLE region).
ADD_FIELD    MSTEP    PORO    FIPNUM3
ADD_FIELD    MSTEP    PERMX   WATER_FLOODED
ADD_FIELD    MSTEP    NTG     MIDDLE 
-------------------------------------------------------------------------------------


    _________________________________________________________________________
   /                                                                         \
   | Observe that prior to loading your hand-crafted configuration file      | 
   | the program will load an ALL_ACTIVE configuration which will be         | 
   | installed as default. If you want you can start with this               |
   | configuration:                                                          |
   |                                                                         |
   | DEL_OBS   ALL_ACTIVE   <Some observation key you do not want to use..>  |
   |                                                                         |
   \_________________________________________________________________________/ 
   



*/
/******************************************************************/


/**
   This file implements the top level object in the system keeping
   track of active/inactive parameters and observations. The system is
   based on three levels.

        1. local_config_type - this implementation

        2. local_updatestep_type - what should be updated and which
           observations to use at one report step.
  
        3. local_ministep_type - what should be updated and which
           observations to use at one enkf update.
           
*/



struct local_config_struct {
  vector_type           * updatestep;            /* This is an indexed vector with (pointers to) local_reportsstep instances. */
  local_updatestep_type * default_updatestep;    /* A default report step returned if no particular report step has been installed for this time index. */
  hash_type             * updatestep_storage;    /* These two hash tables are the 'holding area' for the local_updatestep */
  hash_type             * ministep_storage;      /* and local_ministep instances. */
  stringlist_type       * config_files;
  int                     history_length;
};


static void local_config_clear( local_config_type * local_config ) {
  local_config->default_updatestep  = NULL;
  hash_clear( local_config->updatestep_storage );
  hash_clear( local_config->ministep_storage );
  vector_clear( local_config->updatestep );
  {
    int report;
    for (report=0; report <= local_config->history_length; report++)
      vector_append_ref( local_config->updatestep , NULL );
  }
}




/**
   Observe that history_length is *INCLUSIVE* 
*/
local_config_type * local_config_alloc( int history_length ) {
  local_config_type * local_config = util_malloc( sizeof * local_config , __func__);

  local_config->default_updatestep  = NULL;
  local_config->updatestep_storage  = hash_alloc();
  local_config->ministep_storage    = hash_alloc();
  local_config->updatestep          = vector_alloc_new();
  local_config->history_length      = history_length;
  local_config->config_files = stringlist_alloc_new();
  
  local_config_clear( local_config );
  return local_config;
}


void local_config_free(local_config_type * local_config) {
  vector_free( local_config->updatestep );
  hash_free( local_config->updatestep_storage );
  hash_free( local_config->ministep_storage);
  stringlist_free( local_config->config_files );
  free( local_config );
}



/**
   Actual report step must have been installed in the
   updatestep_storage with local_config_alloc_updatestep() first.
*/

void local_config_set_default_updatestep( local_config_type * local_config , const char * default_key) {
  local_updatestep_type * default_updatestep = local_config_get_updatestep( local_config , default_key );
  local_config->default_updatestep = default_updatestep;
}


/**
   Instances of local_updatestep and local_ministep are allocated from
   the local_config object, and then subsequently manipulated from the calling scope.
*/

local_updatestep_type * local_config_alloc_updatestep( local_config_type * local_config , const char * key ) {
  local_updatestep_type * updatestep = local_updatestep_alloc( key );
  hash_insert_hash_owned_ref( local_config->updatestep_storage , key , updatestep , local_updatestep_free__);
  return updatestep;
}


local_ministep_type * local_config_alloc_ministep( local_config_type * local_config , const char * key ) {
  local_ministep_type * ministep = local_ministep_alloc( key );
  hash_insert_hash_owned_ref( local_config->ministep_storage , key , ministep , local_ministep_free__);
  return ministep;
}


local_ministep_type * local_config_get_ministep( const local_config_type * local_config , const char * key) {
  local_ministep_type * ministep = hash_get( local_config->ministep_storage , key );
  return ministep;
}





local_ministep_type * local_config_alloc_ministep_copy( local_config_type * local_config , const char * src_key , const char * new_key) {
  local_ministep_type * src_step = hash_get( local_config->ministep_storage , src_key );
  local_ministep_type * new_step = local_ministep_alloc_copy( src_step , new_key );
  hash_insert_hash_owned_ref( local_config->ministep_storage , new_key , new_step , local_ministep_free__);
  return new_step;
}



const local_updatestep_type * local_config_iget_updatestep( const local_config_type * local_config , int index) {
  const local_updatestep_type * updatestep = vector_iget_const( local_config->updatestep , index );
  if (updatestep == NULL) 
    /* 
       No particular report step has been installed for this
       time-index, revert to the default.
    */
    updatestep = local_config->default_updatestep;

  if (updatestep == NULL) 
    util_exit("%s: fatal error. No report step information for step:%d - and no default \n",__func__ , index);
    
  return updatestep;
}


local_updatestep_type * local_config_get_updatestep( const local_config_type * local_config , const char * key) {
  return hash_get( local_config->updatestep_storage , key );
}


/**
   This will 'install' the updatestep instance identified with 'key'
   for report steps [step1,step2]. Observe that the report step must
   have been allocated with 'local_config_alloc_updatestep()' first.
*/


void local_config_set_updatestep(local_config_type * local_config, int step1 , int step2 , const char * key) {
  local_updatestep_type * updatestep = hash_get( local_config->updatestep_storage , key );
  int step;
  
  for ( step = step1; step < util_int_min(step2 + 1 , vector_get_size( local_config->updatestep )); step++)
    vector_iset_ref(local_config->updatestep , step , updatestep );
  
}


/*******************************************************************/
/* Functions related to loading a local config instance from disk. */


static local_config_instruction_type local_config_cmd_from_string( hash_type * cmd_table , char * cmd_string ) {
  
  util_strupr( cmd_string );
  if (hash_has_key( cmd_table, cmd_string))
    return hash_get_int( cmd_table , cmd_string);
  else
    util_abort("%s: command:%s not recognized \n",__func__ , cmd_string);
}



const char * local_config_get_cmd_string( local_config_instruction_type cmd ) {
  switch (cmd) {
  case(CREATE_UPDATESTEP):
    return CREATE_UPDATESTEP_STRING;
    break;
  case(CREATE_MINISTEP):
    return CREATE_MINISTEP_STRING;
    break;
  case(ATTACH_MINISTEP):
    return ATTACH_MINISTEP_STRING;
    break;
  case(ADD_DATA):
    return ADD_DATA_STRING;
    break;
  case(ADD_OBS):
    return ADD_OBS_STRING;
    break;
  case(ACTIVE_LIST_ADD_OBS_INDEX):
    return ACTIVE_LIST_ADD_OBS_INDEX_STRING;
    break;
  case(ACTIVE_LIST_ADD_DATA_INDEX):
    return ACTIVE_LIST_ADD_DATA_INDEX_STRING;
    break;
  case(ACTIVE_LIST_ADD_MANY_OBS_INDEX):
    return ACTIVE_LIST_ADD_MANY_OBS_INDEX_STRING;
    break;
  case(ACTIVE_LIST_ADD_MANY_DATA_INDEX):
    return ACTIVE_LIST_ADD_MANY_DATA_INDEX_STRING;
    break;
  case(INSTALL_UPDATESTEP):
    return INSTALL_UPDATESTEP_STRING;
    break;
  case(INSTALL_DEFAULT_UPDATESTEP):
    return INSTALL_DEFAULT_UPDATESTEP_STRING;
    break;
  case(ALLOC_MINISTEP_COPY):
    return ALLOC_MINISTEP_COPY_STRING;
    break;
  case(DEL_DATA):
    return DEL_DATA_STRING;
    break;
  case(DEL_OBS):
    return DEL_OBS_STRING;
    break;
  case(DEL_ALL_DATA):
    return DEL_ALL_DATA_STRING;
    break;
  case(DEL_ALL_OBS):
    return DEL_ALL_OBS_STRING;
    break;
  case(ADD_FIELD):
    return ADD_FIELD_STRING;
    break;
  case(CREATE_REGION):
    return CREATE_REGION_STRING;
    break;
  case(LOAD_FILE):
    return LOAD_FILE_STRING;
    break;
  case(REGION_SELECT_ALL):
    return REGION_SELECT_ALL_STRING;
    break;
  case(REGION_SELECT_VALUE_EQUAL):
    return REGION_SELECT_VALUE_EQUAL_STRING;
    break;
  case(REGION_SELECT_VALUE_LESS):
    return REGION_SELECT_VALUE_LESS_STRING;
    break;
  case(REGION_SELECT_VALUE_MORE):
    return REGION_SELECT_VALUE_MORE_STRING;
    break;
  case(REGION_SELECT_BOX):
    return REGION_SELECT_BOX_STRING;
    break;
  case(REGION_SELECT_SLICE):
    return REGION_SELECT_SLICE_STRING;
    break;
  default:
    util_abort("%s: command:%d not recognized \n",__func__ , cmd);
    return NULL;
  }
}


static int read_int(FILE * stream , bool binary ) {
  if (binary)
    return util_fread_int( stream );
  else {
    int value;
    fscanf(stream , "%d" , &value);
    return value;
  }
}




static void read_int_vector(FILE * stream , bool binary , int_vector_type * vector) {
  if (binary) {
    int size = util_fread_int( stream );
    int_vector_fread_data( vector , size , stream );
  } else {
    int size,value,i;
    int_vector_reset( vector );
    fscanf(stream , "%d" , &size);
    for (i=0; i < size; i++) { 
      if (fscanf(stream , "%d", &value) == 1)
        int_vector_append(vector , value);
      else 
        util_abort("%s: premature end of indices when reading local configuraton - malformed file.\n",__func__);
    }
  }
}


static char * read_alloc_string(FILE * stream , bool binary) {
  if (binary)
    return util_fread_alloc_string( stream );
  else {
    char * string = util_malloc(256 * sizeof * string ,__func__); /* 256 - outht to be enough for everyone ... */
    fscanf(stream , "%s" , string);
    return string;
  }
}


static bool read_bool(FILE * stream , bool binary) {
  if (binary)
    return util_fread_bool( stream );
  else {
    bool value;
    char * s = read_alloc_string( stream , binary );
    if (!util_sscanf_bool( s , &value))
      util_abort("%s: failed to interpret:\'%s\' as boolean true / false\n",__func__ , s );
    free( s );
    return value;
  }
}





static bool read_cmd( hash_type * cmd_table , FILE * stream , bool binary , local_config_instruction_type * cmd) {
  if (binary) { 
    if (fread( cmd , sizeof cmd , 1 , stream) == 1)
      return true;
    else
      return false;
  } else {
    char cmd_string[64];
    if (fscanf(stream , "%s" , cmd_string) == 1) {
      *cmd = local_config_cmd_from_string( cmd_table , cmd_string );
      return true;
    } else
      return false;
  }
}


stringlist_type * local_config_get_config_files( const local_config_type * local_config ) {
  return local_config->config_files;
}


void local_config_clear_config_files( local_config_type * local_config ) {
  stringlist_clear( local_config->config_files );
}


void local_config_add_config_file( local_config_type * local_config , const char * config_file ) {
  stringlist_append_copy( local_config->config_files , config_file );
}



static void local_config_init_cmd_table( hash_type * cmd_table ) {
  hash_insert_int(cmd_table , CREATE_UPDATESTEP_STRING               , CREATE_UPDATESTEP);
  hash_insert_int(cmd_table , CREATE_MINISTEP_STRING                 , CREATE_MINISTEP);
  hash_insert_int(cmd_table , ATTACH_MINISTEP_STRING                 , ATTACH_MINISTEP);
  hash_insert_int(cmd_table , ADD_DATA_STRING                        , ADD_DATA);
  hash_insert_int(cmd_table , ADD_OBS_STRING                         , ADD_OBS );
  hash_insert_int(cmd_table , ACTIVE_LIST_ADD_OBS_INDEX_STRING       , ACTIVE_LIST_ADD_OBS_INDEX);
  hash_insert_int(cmd_table , ACTIVE_LIST_ADD_DATA_INDEX_STRING      , ACTIVE_LIST_ADD_DATA_INDEX);
  hash_insert_int(cmd_table , ACTIVE_LIST_ADD_MANY_OBS_INDEX_STRING  , ACTIVE_LIST_ADD_MANY_OBS_INDEX);
  hash_insert_int(cmd_table , ACTIVE_LIST_ADD_MANY_DATA_INDEX_STRING , ACTIVE_LIST_ADD_MANY_DATA_INDEX);
  hash_insert_int(cmd_table , INSTALL_UPDATESTEP_STRING              , INSTALL_UPDATESTEP);
  hash_insert_int(cmd_table , INSTALL_DEFAULT_UPDATESTEP_STRING      , INSTALL_DEFAULT_UPDATESTEP);
  hash_insert_int(cmd_table , ALLOC_MINISTEP_COPY_STRING             , ALLOC_MINISTEP_COPY);
  hash_insert_int(cmd_table , DEL_DATA_STRING                        , DEL_DATA);
  hash_insert_int(cmd_table , DEL_OBS_STRING                         , DEL_OBS);
  hash_insert_int(cmd_table , DEL_ALL_DATA_STRING                    , DEL_ALL_DATA);
  hash_insert_int(cmd_table , DEL_ALL_OBS_STRING                     , DEL_ALL_OBS);
  hash_insert_int(cmd_table , ADD_FIELD_STRING                       , ADD_FIELD);
  hash_insert_int(cmd_table , CREATE_REGION_STRING                   , CREATE_REGION);
  hash_insert_int(cmd_table , LOAD_FILE_STRING                       , LOAD_FILE);
  hash_insert_int(cmd_table , REGION_SELECT_ALL_STRING               , REGION_SELECT_ALL);
  hash_insert_int(cmd_table , REGION_SELECT_VALUE_EQUAL_STRING       , REGION_SELECT_VALUE_EQUAL);
  hash_insert_int(cmd_table , REGION_SELECT_VALUE_LESS_STRING        , REGION_SELECT_VALUE_LESS);
  hash_insert_int(cmd_table , REGION_SELECT_VALUE_MORE_STRING        , REGION_SELECT_VALUE_MORE);
  hash_insert_int(cmd_table , REGION_SELECT_BOX_STRING               , REGION_SELECT_BOX);
  hash_insert_int(cmd_table , REGION_SELECT_SLICE_STRING             , REGION_SELECT_SLICE);
}


/**
   Currently the ensemble_config and enkf_obs objects are not used for
   anything. These should be used for input validation.
*/

static void local_config_load_file( local_config_type * local_config , const ecl_grid_type * ecl_grid , 
                                    const ensemble_config_type * ensemble_config , const enkf_obs_type * enkf_obs  , 
                                    const char * config_file) {
  bool binary = false;
  local_config_instruction_type cmd;
  hash_type * regions   = hash_alloc();
  hash_type * files     = hash_alloc();
  hash_type * cmd_table = hash_alloc();
  
  FILE * stream       = util_fopen( config_file , "r");
  char * update_name  = NULL;
  char * mini_name    = NULL;
  char * obs_key      = NULL;
  char * data_key     = NULL;
  char * region_name  = NULL;
  int index;
  int_vector_type * int_vector = int_vector_alloc(0,0);
  
  local_config_init_cmd_table( cmd_table );
  while ( read_cmd( cmd_table , stream, binary , &cmd)) {
    switch(cmd) {
    case(CREATE_UPDATESTEP):   
      update_name = read_alloc_string( stream , binary );
      local_config_alloc_updatestep( local_config , update_name );
      break;
    case(CREATE_MINISTEP):
      mini_name = read_alloc_string( stream , binary );
      local_config_alloc_ministep( local_config , mini_name );
      break;
    case(ATTACH_MINISTEP):
      update_name = read_alloc_string( stream , binary );
      mini_name   = read_alloc_string( stream , binary );
      {
        local_updatestep_type * update   = local_config_get_updatestep( local_config , update_name );
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_updatestep_add_ministep( update , ministep );
      }
      break;
    case(ADD_DATA):
      mini_name = read_alloc_string( stream , binary );
      data_key = read_alloc_string( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_add_node( ministep , data_key );
      }
      break;
    case(ADD_OBS):
      mini_name = read_alloc_string( stream , binary );
      obs_key   = read_alloc_string( stream , binary );
      {
        local_ministep_type * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_add_obs( ministep , obs_key );
      }
      break;
    case(ACTIVE_LIST_ADD_OBS_INDEX):
      mini_name = read_alloc_string( stream , binary );
      obs_key   = read_alloc_string( stream , binary );
      index = read_int( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        active_list_type * active_list = local_ministep_get_obs_active_list( ministep , obs_key );
        active_list_add_index( active_list , index );
      }
      break;
    case(ACTIVE_LIST_ADD_DATA_INDEX):
      mini_name = read_alloc_string( stream , binary );
      data_key   = read_alloc_string( stream , binary );
      index      = read_int( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        active_list_type * active_list = local_ministep_get_node_active_list( ministep , data_key );
        active_list_add_index( active_list , index );
      }
      break;
    case(ACTIVE_LIST_ADD_MANY_OBS_INDEX):
      mini_name = read_alloc_string( stream , binary );
      obs_key   = read_alloc_string( stream , binary );
      read_int_vector( stream , binary , int_vector);
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        active_list_type * active_list = local_ministep_get_obs_active_list( ministep , obs_key );
        for (int i = 0; i < int_vector_size( int_vector ); i++) 
          active_list_add_index( active_list , int_vector_iget(int_vector , i));
      }
      break;
    case(ACTIVE_LIST_ADD_MANY_DATA_INDEX):
      mini_name = read_alloc_string( stream , binary );
      data_key   = read_alloc_string( stream , binary );
      read_int_vector( stream , binary , int_vector);
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        active_list_type * active_list = local_ministep_get_node_active_list( ministep , data_key );
        for (int i = 0; i < int_vector_size( int_vector ); i++) 
          active_list_add_index( active_list , int_vector_iget(int_vector , i));
      }
      break;
    case(INSTALL_UPDATESTEP):
      update_name = read_alloc_string( stream , binary );
      {
        int step1,step2;
        
        step1 = read_int( stream , binary );
        step2 = read_int( stream , binary );
        local_config_set_updatestep( local_config , step1 , step2 , update_name );
      }
      break;
    case(INSTALL_DEFAULT_UPDATESTEP):
      update_name = read_alloc_string( stream , binary );
      local_config_set_default_updatestep( local_config , update_name );
      break;
    case(DEL_DATA):
      mini_name = read_alloc_string( stream , binary );
      data_key  = read_alloc_string( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_del_node( ministep , data_key );
      }
      break;
    case(DEL_OBS):
      mini_name = read_alloc_string( stream , binary );
      obs_key   = read_alloc_string( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_del_obs( ministep , obs_key );
      }
      break;
    case(DEL_ALL_DATA):
      mini_name = read_alloc_string( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_clear_nodes( ministep );
      }
      break;
    case(DEL_ALL_OBS):
      mini_name = read_alloc_string( stream , binary );
      {
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_clear_observations( ministep );
      }
      break;
    case(ALLOC_MINISTEP_COPY):
      {
        char * src_name    = read_alloc_string( stream , binary );
        char * target_name = read_alloc_string( stream , binary );
        local_config_alloc_ministep_copy( local_config , src_name , target_name );
        free( src_name );
        free( target_name );
      }
      break;
    case(ADD_FIELD):
      {
        char * field_name;
        mini_name   = read_alloc_string( stream , binary );
        field_name  = read_alloc_string( stream , binary );
        region_name = read_alloc_string( stream , binary );
        {
          ecl_region_type     * region   = hash_get( regions , region_name );
          local_ministep_type * ministep = local_config_get_ministep( local_config , mini_name );
          local_ministep_add_node( ministep , field_name );
          {
            active_list_type * active_list        = local_ministep_get_node_active_list( ministep , field_name );
            const int_vector_type * region_active = ecl_region_get_active_list( region );
            
            for (int i=0; i < int_vector_size( region_active ); i++)
              active_list_add_index( active_list , int_vector_iget( region_active , i ) );
          }
        }
        free( field_name );
      }
      break;
    case(CREATE_REGION):
      {
        region_name = read_alloc_string( stream , binary );
        bool   preselect   = read_bool( stream , binary );
        ecl_region_type * new_region = ecl_region_alloc( ecl_grid , preselect );
        hash_insert_hash_owned_ref( regions , region_name , new_region , ecl_region_free__);
      }
      break;
    case(LOAD_FILE):
      {
        char * file_key  = read_alloc_string( stream , binary ); 
        char * file_name = read_alloc_string( stream , binary );
        ecl_file_type * ecl_file = ecl_file_fread_alloc( file_name );
        hash_insert_hash_owned_ref( files , file_key , ecl_file , ecl_file_free__);
        free( file_key );
        free( file_name );
      }
      break;
    case( REGION_SELECT_BOX ):  /* The coordinates in the box are inclusive in both upper and lower limit,
                                   and the counting starts at 1. */
      {
        region_name = read_alloc_string( stream , binary );
        {
          int i1          = read_int( stream , binary ) - 1;
          int i2          = read_int( stream , binary ) - 1;
          int j1          = read_int( stream , binary ) - 1;
          int j2          = read_int( stream , binary ) - 1;
          int k1          = read_int( stream , binary ) - 1;
          int k2          = read_int( stream , binary ) - 1;
          bool select     = read_bool( stream , binary );
          
          ecl_region_type * region = hash_get( regions , region_name );
          
          if (select)
            ecl_region_select_from_ijkbox( region , i1 , i2 , j1 , j2 , k1 , k2);
          else
            ecl_region_deselect_from_ijkbox( region , i1 , i2 , j1 , j2 , k1 , k2);
          
        }
      }
      break;
    case( REGION_SELECT_SLICE ):
      region_name = read_alloc_string( stream , binary );
      {
        char * dir       = read_alloc_string( stream , binary );
        int n1           = read_int( stream , binary) - 1;
        int n2           = read_int( stream , binary) - 1;
        bool     select = read_bool( stream , binary );
        ecl_region_type * region = hash_get( regions , region_name );
        
        util_strupr( dir );
        
        if (strcmp( dir , "X") == 0) {
          if (select)
            ecl_region_select_i1i2( region , n1 , n2 );
          else
            ecl_region_deselect_i1i2( region , n1 , n2 );
        } else if (strcmp(dir , "Y") == 0) {
          if (select)
            ecl_region_select_j1j2( region , n1 , n2 );
          else
            ecl_region_deselect_j1j2( region , n1 , n2 );
        } else if (strcmp(dir , "Z") == 0) {
          if (select)
            ecl_region_select_k1k2( region , n1 , n2 );
          else
            ecl_region_deselect_k1k2( region , n1 , n2 );
        } else 
          util_abort("%s: slice direction:%s not recognized \n",__func__ , dir );
        
        free(dir );
      }
      break;
    case( REGION_SELECT_ALL ):
      {
        region_name = read_alloc_string( stream , binary );
        bool select = read_bool( stream , binary );
        ecl_region_type * region = hash_get( regions , region_name );
        if (select)
          ecl_region_select_all( region );
        else
          ecl_region_deselect_all( region );
      }
      break;
    case( REGION_SELECT_VALUE_LESS  ):
    case( REGION_SELECT_VALUE_EQUAL ):
    case( REGION_SELECT_VALUE_MORE  ):
      {
        char * master_key;
        ecl_kw_type * ecl_kw;
        ecl_region_type * region;
        char * value_string;
        bool select;

        region_name  = read_alloc_string( stream , binary );
        master_key   = read_alloc_string( stream , binary );
        value_string = read_alloc_string( stream , binary );
        select       = read_bool( stream , binary );
        
        {
          stringlist_type * key_list = stringlist_alloc_from_split( master_key , ":");
          ecl_file_type * ecl_file   = hash_get( files , stringlist_iget(key_list , 0 ));
          int key_nr = 0;
          if (stringlist_get_size( key_list ) == 3) 
            util_sscanf_int( stringlist_iget( key_list , 2 ) , &key_nr );
          
          ecl_kw = ecl_file_iget_named_kw( ecl_file , stringlist_iget( key_list , 1 ) , key_nr);
          stringlist_free( key_list );
        }
        
        region = hash_get( regions , region_name );

        if (cmd == REGION_SELECT_VALUE_EQUAL) {
          int value;
          util_sscanf_int( value_string , &value );
          if (select)
            ecl_region_select_equal( region , ecl_kw , value );
          else
            ecl_region_deselect_equal( region , ecl_kw , value);
        } else {
          double value;
          util_sscanf_double( value_string , &value );

          if (cmd == REGION_SELECT_VALUE_LESS) {
            if (select)
              ecl_region_select_smaller( region , ecl_kw , value );
            else
              ecl_region_deselect_smaller( region , ecl_kw , value);
          } else if (cmd == REGION_SELECT_VALUE_LESS) {
            if (select)
              ecl_region_select_larger( region , ecl_kw , value );
            else
              ecl_region_deselect_larger( region , ecl_kw , value);
          }

        }
        free( master_key );
        free( value_string );
      }
      break;
    default:
      util_abort("%s: invalid command:%d \n",__func__ , cmd);
    }

    util_safe_free( region_name );  region_name = NULL;
    util_safe_free( update_name );  update_name = NULL;
    util_safe_free( mini_name );    mini_name   = NULL;
    util_safe_free( obs_key );      obs_key     = NULL;
    util_safe_free( data_key );     data_key    = NULL;
  }
  fclose(stream);
  int_vector_free( int_vector );
  hash_free( regions );
  hash_free( files );
  hash_free( cmd_table );
}



/*
  Should probably have a "modified" flag to ensure internal consistency 
*/

void local_config_reload( local_config_type * local_config , const ecl_grid_type * ecl_grid , const ensemble_config_type * ensemble_config , const enkf_obs_type * enkf_obs  , const char * all_active_config_file ) {
  local_config_clear( local_config );
  if (all_active_config_file != NULL)
    local_config_load_file( local_config , ecl_grid , ensemble_config , enkf_obs , all_active_config_file );
  {
    int i;
    for (i = 0; i < stringlist_get_size( local_config->config_files ); i++)
      local_config_load_file( local_config , ecl_grid , ensemble_config , enkf_obs , stringlist_iget( local_config->config_files , i ) );
  }
}



void local_config_fprintf( const local_config_type * local_config , const char * config_file) {
  FILE * stream = util_mkdir_fopen( config_file , "w");
  
  /* Start with dumping all the ministep instances. */
  {
    hash_iter_type * hash_iter = hash_iter_alloc( local_config->ministep_storage );

    while (!hash_iter_is_complete( hash_iter )) {
      const local_ministep_type * ministep = hash_iter_get_next_value( hash_iter );
      local_ministep_fprintf( ministep , stream );
    }
        
    hash_iter_free( hash_iter );
  }
  
  
  /* Dumping all the reportstep instances as ATTACH_MINISTEP commands. */
  {
    hash_iter_type * hash_iter = hash_iter_alloc( local_config->updatestep_storage );
    
    while (!hash_iter_is_complete( hash_iter )) {
      const local_updatestep_type * updatestep = hash_iter_get_next_value( hash_iter );
      local_updatestep_fprintf( updatestep , stream );
    }
        
    hash_iter_free( hash_iter );
  }

  /* Writing out the updatestep / time */
  {
    int i;
    for (i=0; i < vector_get_size( local_config->updatestep ); i++) {
      const local_updatestep_type * updatestep = vector_iget_const( local_config->updatestep , i );
      if (updatestep != NULL)
        fprintf(stream , "%s %s %d %d \n", local_config_get_cmd_string( INSTALL_UPDATESTEP ) , local_updatestep_get_name( updatestep ) , i , i );
    }
  }

  /* Installing the default updatestep */
  if (local_config->default_updatestep != NULL) 
    fprintf(stream , "%s %s\n", local_config_get_cmd_string( INSTALL_DEFAULT_UPDATESTEP ) , local_updatestep_get_name( local_config->default_updatestep ));
  
  fclose( stream );
}



void local_config_fprintf_config( const local_config_type * local_config , FILE * stream) {
  fprintf( stream , CONFIG_COMMENTLINE_FORMAT );
  fprintf( stream , CONFIG_COMMENT_FORMAT , "Here comes the config files used for setting up local analysis.");
  for (int i=0; i < stringlist_get_size( local_config->config_files ); i++) {
    fprintf(stream , CONFIG_KEY_FORMAT      , LOCAL_CONFIG_KEY );
    fprintf(stream , CONFIG_ENDVALUE_FORMAT , stringlist_iget( local_config->config_files , i ));
  }
}
