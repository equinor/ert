#ifndef __MEAS_DATA_H__
#define __MEAS_DATA_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <matrix.h>
#include <hash.h>

typedef struct meas_data_struct meas_data_type;
typedef struct meas_block_struct  meas_block_type;  

void               meas_block_iset( meas_block_type * meas_block , int iens , int iobs , double value);
double             meas_block_iget_ens_mean( const meas_block_type * meas_block , int iobs );
double             meas_block_iget_ens_std( const meas_block_type * meas_block , int iobs);
void               meas_block_deactivate( meas_block_type * meas_block , int iobs );                               

void meas_data_fprintf( const meas_data_type * matrix , FILE * stream);

void               meas_data_reset(meas_data_type * );
meas_data_type * meas_data_alloc( int );
void               meas_data_free(meas_data_type * );
void               meas_data_add(meas_data_type * , int , double );
matrix_type      * meas_data_allocS(const meas_data_type * matrix , int active_size);
void               meas_data_deactivate(meas_data_type * meas_data, int index);
double             meas_data_iget_ens_mean(const meas_data_type * matrix , int index);
double             meas_data_iget_ens_std(const meas_data_type * matrix , int index);
void               meas_data_iget_ens_mean_std( const meas_data_type * matrix , int index , double * mean , double * std);
int                meas_data_get_ens_size( const meas_data_type * meas_data );
int                meas_data_get_nrobs( const meas_data_type * meas_data );
meas_block_type  * meas_data_add_block( meas_data_type * matrix , const char * obs_key , int report_step , int obs_size);
meas_block_type  * meas_data_iget_block( meas_data_type * matrix , int block_mnr);
const meas_block_type  * meas_data_iget_block_const( const meas_data_type * matrix , int block_nr );
void               meas_block_calculate_ens_stats( meas_block_type * meas_block );
int                meas_block_get_total_size( const meas_block_type * meas_block );
bool               meas_block_iget_active( const meas_block_type * meas_block , int iobs);
void               meas_data_assign_vector(meas_data_type * target_matrix, const meas_data_type * src_matrix , int target_index , int src_index);
meas_data_type   * meas_data_alloc_copy( const meas_data_type * src );

#ifdef __cplusplus
}
#endif
#endif
