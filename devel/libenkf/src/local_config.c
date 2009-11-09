#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector.h>
#include <util.h>
#include <local_ministep.h>
#include <local_updatestep.h>
#include <local_config.h>
#include <int_vector.h>
#include <log.h>
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
This function will delete all the obs keys from the ministep
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
   



The format of this config file does not support any form of comments,
but it is completely ignorant about whitespace (including blank
lines).

Small example:
--------------
CREATE_UPDATESTEP                UPDATE
CREATE_MINISTEP                  MINI
ATTACH_MINISTEP                  UPDATE MINI
ADD_OBS                          MINI   WWCT:OP_1
ADD_OBS                          MINI   WGOR:OP_1
ADD_OBS                          MINI   RFT3
ADD_DATA                         MINI   FLUID_PARAMS
ADD_DATA                         MINI   GRID_PARAMS
ADD_DATA                         MINI   PRESSURE
ACTIVE_LIST_ADD_MANY_DATA_INDEX  MINI   PRESSURE 10 0 1 2 3 4 5 10 20 30 40
INSTALL_DEFAULT_UPDATESTEP       UPDATE
---------------------------------------------------------------------------





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
  hash_type 		* updatestep_storage;    /* These two hash tables are the 'holding area' for the local_updatestep */
  hash_type 		* ministep_storage;      /* and local_ministep instances. */
};



/**
   Observe that history_length is *INCLUSIVE* 
*/
local_config_type * local_config_alloc( int history_length ) {
  local_config_type * local_config = util_malloc( sizeof * local_config , __func__);

  local_config->default_updatestep = NULL;
  local_config->updatestep_storage  = hash_alloc();
  local_config->ministep_storage    = hash_alloc();
  local_config->updatestep          = vector_alloc_new();
  {
    int report;
    for (report=0; report <= history_length; report++)
      vector_append_ref( local_config->updatestep , NULL );
  }
  
  return local_config;
}


void local_config_free(local_config_type * local_config) {
  vector_free( local_config->updatestep );
  hash_free( local_config->updatestep_storage );
  hash_free( local_config->ministep_storage);
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
  if (updatestep == NULL) {
    /* 
       No particular report step has been installed for this
       time-index, revert to the default.
    */
    updatestep = local_config->default_updatestep;
    printf("Returning default update step \n");
  }

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


static local_config_instruction_type local_config_cmd_from_string( char * cmd_string ) {
  local_config_instruction_type cmd;

  util_strupr( cmd_string );
  if (strcmp( cmd_string , CREATE_UPDATESTEP_STRING) == 0)
    cmd = CREATE_UPDATESTEP;
  else if (strcmp( cmd_string , CREATE_MINISTEP_STRING) == 0)
    cmd = CREATE_MINISTEP;
  else if (strcmp( cmd_string , ATTACH_MINISTEP_STRING) == 0)
    cmd =  ATTACH_MINISTEP;
  else if (strcmp( cmd_string , ADD_DATA_STRING) == 0)
    cmd = ADD_DATA;
  else if (strcmp( cmd_string , ADD_OBS_STRING) == 0)
    cmd = ADD_OBS;
  else if (strcmp( cmd_string , ACTIVE_LIST_ADD_OBS_INDEX_STRING) == 0)
    cmd = ACTIVE_LIST_ADD_OBS_INDEX;
  else if (strcmp( cmd_string , ACTIVE_LIST_ADD_DATA_INDEX_STRING) == 0)
    cmd = ACTIVE_LIST_ADD_DATA_INDEX;
  else if (strcmp( cmd_string , ACTIVE_LIST_ADD_MANY_OBS_INDEX_STRING) == 0)
    cmd = ACTIVE_LIST_ADD_MANY_OBS_INDEX;
  else if (strcmp( cmd_string , ACTIVE_LIST_ADD_MANY_DATA_INDEX_STRING) == 0)
    cmd = ACTIVE_LIST_ADD_MANY_DATA_INDEX;
  else if (strcmp( cmd_string , INSTALL_UPDATESTEP_STRING) == 0)
    cmd = INSTALL_UPDATESTEP;
  else if (strcmp( cmd_string , INSTALL_DEFAULT_UPDATESTEP_STRING) == 0)
    cmd = INSTALL_DEFAULT_UPDATESTEP;
  else if (strcmp( cmd_string , ALLOC_MINISTEP_COPY_STRING) == 0)
    cmd = ALLOC_MINISTEP_COPY;
  else if (strcmp( cmd_string , DEL_DATA_STRING) == 0)
    cmd = DEL_DATA;
  else if (strcmp( cmd_string , DEL_OBS_STRING) == 0)
    cmd = DEL_OBS;
  else if (strcmp( cmd_string ,DEL_ALL_DATA_STRING ) == 0)
    cmd = DEL_ALL_DATA;
  else {
    util_abort("%s: Command:%s not recognized \n",__func__ , cmd_string);
    cmd = INVALID_CMD;
  }
  return cmd;
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
      fscanf(stream , "%d", &value);
      int_vector_append(vector , value);
    }
  }
}


static char * read_alloc_string(FILE * stream , bool binary) {
  if (binary)
    return util_fread_alloc_string( stream );
  else {
    char * string = util_malloc(64 * sizeof * string ,__func__); /*64 - outht to be enough for everyone ... */
    fscanf(stream , "%s" , string);
    return string;
  }
}



static bool read_cmd( FILE * stream , bool binary , local_config_instruction_type * cmd) {
  if (binary) { 
    if (fread( cmd , sizeof cmd , 1 , stream) == 1)
      return true;
    else
      return false;
  } else {
    char cmd_string[64];
    if (fscanf(stream , "%s" , cmd_string) == 1) {
      *cmd = local_config_cmd_from_string( cmd_string );
      return true;
    } else
      return false;
  }
}


void local_config_load( local_config_type * local_config /*const ensemble_config_type * ensemble_config , const enkf_obs_type * enkf_obs , */ , const char * config_file , log_type * logh) {
  bool binary = false;
  local_config_instruction_type cmd;
  FILE * stream      = util_fopen( config_file , "r");
  char * update_name = NULL;
  char * mini_name   = NULL;
  char * obs_key     = NULL;
  char * data_key    = NULL;
  int index;
  int_vector_type * int_vector = int_vector_alloc(0,0);

  log_add_fmt_message(logh , 1 , NULL , "Loading local configuration from file:%s" , config_file);
  while ( read_cmd(stream, binary , &cmd)) {
    switch(cmd) {
    case(CREATE_UPDATESTEP):   
      update_name = read_alloc_string( stream , binary );
      local_config_alloc_updatestep( local_config , update_name );
      log_add_fmt_message(logh , 2 , NULL , "Added local update step:%s" , update_name);
      break;
    case(CREATE_MINISTEP):
      mini_name = read_alloc_string( stream , binary );
      local_config_alloc_ministep( local_config , mini_name );
      log_add_fmt_message(logh , 2 , NULL , "Added local mini step:%s" , mini_name);
      break;
    case(ATTACH_MINISTEP):
      update_name = read_alloc_string( stream , binary );
      mini_name   = read_alloc_string( stream , binary );
      {
        local_updatestep_type * update   = local_config_get_updatestep( local_config , update_name );
        local_ministep_type   * ministep = local_config_get_ministep( local_config , mini_name );
        log_add_fmt_message(logh , 2 , NULL , "Attached ministep:%s to update_step:%s", mini_name , update_name);
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
      log_add_fmt_message(logh , 3 , NULL , "Added data:%s to mini step:%s" , data_key , mini_name);
      break;
    case(ADD_OBS):
      mini_name = read_alloc_string( stream , binary );
      obs_key   = read_alloc_string( stream , binary );
      {
        local_ministep_type * ministep = local_config_get_ministep( local_config , mini_name );
        local_ministep_add_obs( ministep , obs_key );
      }
      log_add_fmt_message(logh , 3 , NULL , "Added observation:%s to mini step:%s" , obs_key , mini_name);
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
      index = read_int( stream , binary );
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
      }
      break;
    default:
      util_abort("%s: invalid command:%d \n",__func__ , cmd);
    }
    
    update_name = util_safe_free( update_name );
    mini_name   = util_safe_free( mini_name );
    obs_key     = util_safe_free( obs_key );
    data_key    = util_safe_free( data_key );
  }
  fclose(stream);
  int_vector_free( int_vector );
}
