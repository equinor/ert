#ifndef __PETP_H__
#define __PETP_H__
#include <hash.h>

typedef struct petp_struct petp_type;

petp_type * petp_fscanf_alloc(const char *, const hash_type *);
void petp_free(petp_type *);
void petp_fwrite(const petp_type *, const int *, const int *, int, const char *, bool); 

#endif
