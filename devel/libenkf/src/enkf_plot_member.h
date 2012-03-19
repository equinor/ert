/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_plot_member.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __ENKF_PLOT_MEMBER_H__
#define __ENKF_PLOT_MEMBER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>
#include <member_config.h>
  
  typedef struct enkf_plot_member_struct enkf_plot_member_type;
  
  enkf_plot_member_type * enkf_plot_member_alloc( const member_config_type * member_config , time_t start_time);
  void                    enkf_plot_member_load( enkf_plot_member_type * plot_member , enkf_node_type * enkf_node , enkf_fs_type * fs , const char * user_key , state_enum state , int step1 , int step2);
  void                    enkf_plot_member_load__( void *arg );
  void                    enkf_plot_member_free__( void * arg );

#ifdef __cplusplus
}
#endif
#endif
