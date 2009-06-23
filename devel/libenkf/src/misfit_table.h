#ifndef __MISFIT_TABLE_H__
#define __MISFIT_TABLE_H__
#include <enkf_obs.h>

typedef struct misfit_table_struct misfit_table_type;



misfit_table_type * misfit_table_alloc( int history_length , int ens_size , const enkf_obs_type * enkf_obs );
void                misfit_table_free( misfit_table_type * table );

#endif
