#ifndef __TPGZONE_TRUNC_SCHEME_H__
#define __TPGZONE_TRUNC_SCHEME_H__

#include <void_arg.h>

typedef int (tpgzone_trunc_scheme_func_type)(const double *,int, void_arg_type *);

typedef struct tpgzone_trunc_scheme_node_type_struct tpgzone_trunc_scheme_node_type;
typedef struct tpgzone_trunc_scheme_type_struct tpgzone_trunc_scheme_type;

int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *, int, const double *);
int tpgzone_trunc_scheme_type_apply(tpgzone_trunc_scheme_type *,const double *);

tpgzone_trunc_scheme_type * tpgzone_trunc_scheme_type_fscanf_alloc(const char *);
#endif
