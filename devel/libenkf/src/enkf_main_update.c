#include <matrix.h>
#include <local_config.h>
#include <local_ministep.h>
#include <local_reportstep.h>
#include <enkf_node.h>
#include <enkf_state.h>
#include <enkf_analysis.h>
#include <analysis_config.h>

/**
   This file implements functions related to updating the state. The
   really low level functions related to the analysis is implemented
   in enkf_analysis.h.
   
   This file only implements code for updating, and not any data
   structures.
*/

#include "enkf_main_struct.h"


