/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_plot_blockdata.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __ENKF_PLOT_BLOCKDATA_H__
#define __ENKF_PLOT_BLOCKDATA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>

#include <ert/enkf/obs_vector.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_plot_blockvector.h>


  typedef struct enkf_plot_blockdata_struct enkf_plot_blockdata_type;
  
  enkf_plot_blockdata_type * enkf_plot_blockdata_alloc( const obs_vector_type * obs_vector );
  void enkf_plot_blockdata_free( enkf_plot_blockdata_type * data );
  int  enkf_plot_blockdata_get_size( const enkf_plot_blockdata_type * data );
  enkf_plot_blockvector_type * enkf_plot_blockdata_iget( const enkf_plot_blockdata_type * plot_data , int index);
  void enkf_plot_blockdata_load( enkf_plot_blockdata_type * plot_data , 
                                 enkf_fs_type * fs , 
                                 int report_step , 
                                 state_enum state , 
                                 const bool_vector_type * input_mask);

  UTIL_IS_INSTANCE_HEADER( enkf_plot_blockdata );

#ifdef __cplusplus
}
#endif
#endif
