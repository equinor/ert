#ifndef __TRANS_FUNC_H__
#define __TRANS_FUNC_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_types.h>
#include <arg_pack.h>
#include <stdio.h>
#include <stdbool.h>


typedef struct trans_func_struct  trans_func_type;
typedef double (transform_ftype) (double , const arg_pack_type *);
typedef bool   (validate_ftype)  (const trans_func_type * );

trans_func_type  * trans_func_fscanf_alloc( FILE * stream );
double             trans_func_eval( const trans_func_type * trans_func , double x);

void               trans_func_free( trans_func_type * trans_func );
void               trans_func_iset_double_param(trans_func_type  * trans_func , int param_index , double value );
bool               trans_func_set_double_param( trans_func_type  * trans_func , const char * param_name , double value );
void               trans_func_iset_int_param(trans_func_type  * trans_func , int param_index , int value );
bool               trans_func_set_int_param( trans_func_type  * trans_func , const char * param_name , int value );


#ifdef __cplusplus
}
#endif
#endif
