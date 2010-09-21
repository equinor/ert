#ifndef __MEAS_MATRIX_H__
#define __MEAS_MATRIX_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <matrix.h>
#include <hash.h>

typedef struct meas_matrix_struct meas_matrix_type;
typedef struct meas_block_struct  meas_block_type;  

void               meas_block_iset( meas_block_type * meas_block , int iens , int iobs , double value);
double             meas_block_iget_ens_mean( const meas_block_type * meas_block , int iobs );
double             meas_block_iget_ens_std( const meas_block_type * meas_block , int iobs);
void               meas_block_deactivate( meas_block_type * meas_block , int iobs );                               

void meas_matrix_fprintf( const meas_matrix_type * matrix , FILE * stream);

void               meas_matrix_reset(meas_matrix_type * );
meas_matrix_type * meas_matrix_alloc( int );
void               meas_matrix_free(meas_matrix_type * );
void               meas_matrix_add(meas_matrix_type * , int , double );
matrix_type      * meas_matrix_allocS(const meas_matrix_type * matrix , int active_size);
void               meas_matrix_deactivate(meas_matrix_type * meas_matrix, int index);
double             meas_matrix_iget_ens_mean(const meas_matrix_type * matrix , int index);
double             meas_matrix_iget_ens_std(const meas_matrix_type * matrix , int index);
void               meas_matrix_iget_ens_mean_std( const meas_matrix_type * matrix , int index , double * mean , double * std);
int                meas_matrix_get_ens_size( const meas_matrix_type * meas_matrix );
int                meas_matrix_get_nrobs( const meas_matrix_type * meas_matrix );
meas_block_type  * meas_matrix_add_block( meas_matrix_type * matrix , const char * obs_key , int report_step , int obs_size);
meas_block_type  * meas_matrix_iget_block( meas_matrix_type * matrix , int block_mnr);
const meas_block_type  * meas_matrix_iget_block_const( const meas_matrix_type * matrix , int block_nr );
void               meas_block_calculate_ens_stats( meas_block_type * meas_block );
int                meas_block_get_total_size( const meas_block_type * meas_block );
bool               meas_block_iget_active( const meas_block_type * meas_block , int iobs);
void               meas_matrix_assign_vector(meas_matrix_type * target_matrix, const meas_matrix_type * src_matrix , int target_index , int src_index);
meas_matrix_type * meas_matrix_alloc_copy( const meas_matrix_type * src );

#ifdef __cplusplus
}
#endif
#endif
