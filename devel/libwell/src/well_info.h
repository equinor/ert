/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'well_info.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __WELL_INFO_H__
#define __WELL_INFO_H__


#ifdef __cplusplus
extern "C" {
#endif

#include <ecl_file.h>


  typedef struct well_info_struct well_info_type;
  
  well_info_type *  well_info_alloc();
  void              well_info_add_UNRST_wells( well_info_type * well_info , ecl_file_type * rst_file, int grid_nr);
  void              well_info_add_wells( well_info_type * well_info , ecl_file_type * rst_file , int report_nr , int grid_nr);
  void              well_info_load_file( well_info_type * well_info , const char * filename);
  void              well_info_free( well_info_type * well_info );
  well_state_type * well_info_get_state_from_time( const well_info_type * well_info , const char * well_name , time_t sim_time);
  well_state_type * well_info_get_state_from_report( const well_info_type * well_info , const char * well_name , int report_step );
  well_state_type * well_info_iget_state_from_report( const well_info_type * well_info , const char * well_name , int index);
  int               well_info_get_well_size( const well_info_type * well_info , const char * well_name );

#ifdef __cplusplus
}
#endif

#endif
