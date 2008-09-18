#ifndef __GRUPTREE_H__
#define __GRUPTREE_H__

#include <hash.h>

typedef struct gruptree_struct gruptree_type;

gruptree_type * gruptree_alloc_emtpy();
void            gruptree_free(gruptree_type *);
void            gruptree_register_grup(gruptree_type *, const char *, const char *);
void            gruptree_register_well(gruptree_type *, const char *, const char *);
char         ** gruptree_alloc_well_names(gruptree_type *, const char *, int *);

void            gruptree_printf_wells(gruptree_type *, const char *);
#endif
