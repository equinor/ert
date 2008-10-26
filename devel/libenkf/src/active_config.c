#include <stdlib.h>
#include <util.h>
#include <enkf_types.h>
#include <ensemble_config.h>
#include <enkf_obs.h>
#include <active_config.h>
#include <active_node.h>
#include <hash.h>

/**
   This file implements the top level object in the system keeping
   track of active/inactive parameters and observations. The system is
   based on three levels, from the bottom up:

        1. active_map_type

        2. active_report_step_type
  
        3. active_type


   active_map_type 
   --------------- 
   This object contains all the information about active/inactive
   parameters and observations for _one_single_enkf_update_. This is
   many respects the most importan object in the active/inactive
   system.

   In many cases there will be only one active_map instance active at
   one report_step, but when some form of local analysis is applied,
   there wille be several active_map instances at the same
   report_step; one for each local update.
                  
                 
   active_report_step_type 
   -----------------------
   This is active/inactive information for _one_ report step. The main
   content is a list of active_map_type instances. In the case of
   global analyis, this will just conists of one element.


   active_type
   -----------
   This contains 

*/

struct active_map_struct {
  int __id;
};
 
