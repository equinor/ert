#include <stdlib.h>
#include <util.h>
#include <enkf_types.h>
#include <enkf_config_node.h>
#include <ensemble_config.h>
#include <active_node.h>
#include <obs_node.h>

/**
   This file implements the two most basic objects used in the mapping
   of active/inactive observations and variables. One of these nodes
   contains the information necessary to activate/deactivate one
   variable/observation.
*/


/**
   This struct implements the holding information for the
   activation/deactivation of one variable.
*/
  
struct active_var_struct {
  const enkf_config_node_type    * config_node;          /* The enkf_config_node instance this is all about - pointer to *shared* resource. */
  active_mode_type                 active_mode;         
  void                      	 * active_config;        /* An object (type depending on datatype of config_node) used to hold info abourt partly active variable. 
			    	 			    Owned by this object. If active_mode == all_active or active_mode == inactive, this can be NULL. */
  active_config_destructor_ftype * free_active_config;   /* Destructor for the active_config object, can be NULL if that object is NULL. */
};



/**
   Similar to active_var_struct, but for observations.
*/
struct active_obs_struct {
  const obs_node_type       	  * obs_node;             /* The obs_node instance this is all about - pointer to *shared* resource. */
  active_mode_type          	    active_mode;         
  void                      	  * active_config;        /* An object (type depending on datatype of obs_node) used to hold info abourt partly active variable. 
                   					     Owned by this object. If active_mode == all_active or active_mode == inactive, this can be NULL. */
  active_config_destructor_ftype  * free_active_config;   /* Destructor for the active_config object, can be NULL if that object is NULL. */
};




void active_node_
