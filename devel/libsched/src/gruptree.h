#ifndef __GRUPTREE_H__
#define __GRUPTREE_H__

#include <stdio.h>
#include <hash.h>

typedef struct gruptree_struct gruptree_type;

gruptree_type * gruptree_alloc();
gruptree_type * gruptree_copyc(const gruptree_type *);
gruptree_type * gruptree_fread_alloc(FILE *);
void            gruptree_fwrite(const gruptree_type *, FILE *);
void            gruptree_free(gruptree_type *);


void            gruptree_register_grup(gruptree_type *, const char *, const char *);
void            gruptree_register_well(gruptree_type *, const char *, const char *);
bool            gruptree_has_grup(const gruptree_type *, const char *);
char         ** gruptree_alloc_grup_well_list(gruptree_type *, const char *, int *);
void            gruptree_printf_grup_wells(gruptree_type *, const char *);

#endif
