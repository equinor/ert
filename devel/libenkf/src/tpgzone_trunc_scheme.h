#ifndef __TPGZONE_TRUNC_SCHEME_H__
#define __TPGZONE_TRUNC_SCHEME_H__

#include <stdbool.h>

#include <hash.h>

typedef bool (tpgzone_trunc_scheme_func_type)(const double *, const double *, int);

typedef struct tpgzone_trunc_scheme_node_type_struct tpgzone_trunc_scheme_node_type;
typedef struct tpgzone_trunc_scheme_type_struct tpgzone_trunc_scheme_type;

int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *, int, const double *, int);
int tpgzone_trunc_scheme_type_apply(tpgzone_trunc_scheme_type *,const double *);

tpgzone_trunc_scheme_type * tpgzone_trunc_scheme_type_fscanf_alloc(const char *, const hash_type *, int);
#endif
