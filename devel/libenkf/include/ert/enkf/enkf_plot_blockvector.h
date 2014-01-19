/*
   Copyright (C) 2014  Statoil ASA, Norway. 
    
   The file 'enkf_plot_blockvector.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __ENKF_PLOT_BLOCKVECTOR_H__
#define __ENKF_PLOT_BLOCKVECTOR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>

#include <ert/enkf/block_obs.h>

  typedef struct enkf_plot_blockvector_struct enkf_plot_blockvector_type;
  
  enkf_plot_blockvector_type * enkf_plot_blockvector_alloc( const block_obs_type * block_obs );
  void enkf_plot_blockvector_free( enkf_plot_blockvector_type * vector );
  
  UTIL_IS_INSTANCE_HEADER( enkf_plot_blockvector );

#ifdef __cplusplus
}
#endif
#endif
