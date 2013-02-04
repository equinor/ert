/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'analysis_iterated_config.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __ANALYSIS_ITERATED_CONFIG_H__
#define __ANALYSIS_ITERATED_CONFIG_H__

#ifdef __cplusplus 
extern "C" {
#endif


typedef struct analysis_iterated_config_struct analysis_iterated_config_type;

  void                            analysis_iterated_config_set_num_iterations( analysis_iterated_config_type * config , int num_iterations);
  int                             analysis_iterated_config_get_num_iterations( const analysis_iterated_config_type * config );
  analysis_iterated_config_type * analysis_iterated_config_alloc();
  void                            analysis_iterated_config_free( analysis_iterated_config_type * config );
  const char *                    analysis_iterated_config_iget_runpath_fmt( analysis_iterated_config_type * config , int iter);
  const char *                    analysis_iterated_config_iget_enspath_fmt( analysis_iterated_config_type * config , int iter);
  void                            analysis_iterated_config_add_config_items( config_type * config );
  void                            analysis_iterated_config_init(analysis_iterated_config_type * iter_config , const config_type * config);

#ifdef __cplusplus 
}
#endif
#endif
