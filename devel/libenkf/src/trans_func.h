#ifndef __TRANS_FUNC_H__
#define __TRANS_FUNC_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_types.h>
#include <arg_pack.h>
#include <stdio.h>
#include <stdbool.h>

double             trans_errf    (double  , const arg_pack_type *);
double             trans_tanh    (double  , const arg_pack_type *);
double             trans_exp     (double  , const arg_pack_type *);
double             trans_pow10   (double  , const arg_pack_type *);
double             trans_step    (double  , const arg_pack_type *);
double             trans_const   (double  , const arg_pack_type *);
double             trans_normal  (double  , const arg_pack_type *);
double             trans_unif    (double  , const arg_pack_type *);
double             trans_logunif (double  , const arg_pack_type *);
transform_ftype  * trans_func_lookup(FILE * stream, char ** , arg_pack_type ** );

#ifdef __cplusplus
}
#endif
#endif
