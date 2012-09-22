/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'misfit_table.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __MISFIT_TABLE_H__
#define __MISFIT_TABLE_H__

#include <enkf_obs.h>
#include <ensemble_config.h>
#include <enkf_fs.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MISFIT_DEFAULT_RANKING_KEY "DEFAULT"

typedef struct misfit_table_struct misfit_table_type;

  const int         * misfit_table_get_ranking_permutation( const misfit_table_type * table , const char * ranking_key );
  misfit_table_type * misfit_table_alloc( const ensemble_config_type * config , enkf_fs_type * fs , int history_length , int ens_size , const enkf_obs_type * enkf_obs );
  void                misfit_table_free( misfit_table_type * table );
  void                misfit_table_create_ranking(misfit_table_type * table , const stringlist_type * sort_keys , int step1 , int step2, const char * ranking_key , const char * filename);
  void                misfit_table_create_data_ranking(misfit_table_type * table , enkf_fs_type * fs, int ens_size , enkf_config_node_type * config_node, const char * user_key , const char * key_index , int step , state_enum state , const char * ranking_key , const char * filename);
  bool                misfit_table_has_ranking( const misfit_table_type * table , const char * ranking_key );
  void                misfit_table_display_ranking( const misfit_table_type * table , const char * ranking_key );
  void                misfit_table_fwrite( const misfit_table_type * misfit_table , FILE * stream);

#ifdef __cplusplus
}
#endif

#endif
