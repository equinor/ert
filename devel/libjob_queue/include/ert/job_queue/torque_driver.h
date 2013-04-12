/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'torque_driver.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#ifndef TORQUE_DRIVER_H
#define	TORQUE_DRIVER_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>

/*
  The options supported by the Torque driver.
*/
#define TORQUE_QSUB_CMD     "QSUB_CMD"
#define TORQUE_QSTAT_CMD    "QSTAT_CMD"
#define TORQUE_QDEL_CMD    "QDEL_CMD"

typedef struct torque_driver_struct torque_driver_type;

void          * torque_driver_alloc( );
const  void   * torque_driver_get_option( const void * __driver , const char * option_key);
bool            torque_driver_set_option( void * __driver , const char * option_key , const void * value);

UTIL_SAFE_CAST_HEADER( torque_driver );
 
#ifdef	__cplusplus
}
#endif

#endif	/* TORQUE_DRIVER_H */

