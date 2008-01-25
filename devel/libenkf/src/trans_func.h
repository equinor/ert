#ifndef __TRANS_FUNC_H__
#define __TRANS_FUNC_H__
#include <enkf_types.h>
#include <void_arg.h>
#include <stdio.h>
#include <stdbool.h>

double             trans_errf   (double  , const void_arg_type *);
double             trans_tanh   (double  , const void_arg_type *);
double             trans_exp    (double  , const void_arg_type *);
double             trans_pow10  (double  , const void_arg_type *);
double             trans_step   (double  , const void_arg_type *);
double             trans_const  (double  , const void_arg_type *);
double             trans_normal (double  , const void_arg_type *);
double             trans_unif   (double  , const void_arg_type *);
transform_ftype  * trans_func_lookup(FILE * stream, char ** , void_arg_type ** , bool *);

#endif
