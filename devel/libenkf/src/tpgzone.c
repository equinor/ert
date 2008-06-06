#include <tpgzone.h>


#define DEBUG
#include "enkf_debug.h"


/*
  Implementation of the tpgzone_struct
*/
struct tpgzone_struct
{
  DEBUG_DECLARE
  tpgzone_config_type * config;
  double              * data;
};
