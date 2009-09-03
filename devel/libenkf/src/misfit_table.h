#ifndef __MISFIT_TABLE_H__
#define __MISFIT_TABLE_H__
#include <enkf_obs.h>
#include <ensemble_config.h>
#include <enkf_fs.h>

#define MISFIT_DEFAULT_RANKING_KEY "DEFAULT"

typedef struct misfit_table_struct misfit_table_type;

const int         * misfit_table_get_ranking_permutation( const misfit_table_type * table , const char * ranking_key );
misfit_table_type * misfit_table_alloc( const ensemble_config_type * config , enkf_fs_type * fs , int history_length , int ens_size , const enkf_obs_type * enkf_obs );
void                misfit_table_free( misfit_table_type * table );
void                misfit_table_create_ranking(misfit_table_type * table , const stringlist_type * sort_keys , int step1 , int step2, const char * ranking_key , const char * filename);
bool                misfit_table_has_ranking( const misfit_table_type * table , const char * ranking_key );
void                misfit_table_display_ranking( const misfit_table_type * table , const char * ranking_key );
void                misfit_table_fwrite( const misfit_table_type * misfit_table , FILE * stream);

#endif
