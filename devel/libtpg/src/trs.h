#ifndef __TRS_H__
#define __TRS_H__

#include <hash.h>

typedef struct trs_type_struct trs_type;
int trs_apply_block(trs_type *,const double *);
int * trs_apply_alloc(trs_type *, const double **, int);
trs_type * trs_fscanf_alloc(const char *, const hash_type *, int);

void trs_free(trs_type *);

#endif
