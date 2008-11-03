#ifndef __CFG_STRUCT_H__
#define __CFG_STRUCT_H__

#include <cfg_struct_def.h>
#include <stringlist.h>

typedef struct cfg_struct_struct cfg_struct_type;

// Manipulators
cfg_struct_type * cfg_struct_alloc_from_file(const char *, const cfg_struct_def_type *);
void              cfg_struct_free(cfg_struct_type *);
void              cfg_struct_free__(void *);


// Accessors
const char *      cfg_struct_get_name(const cfg_struct_type *);
const char *      cfg_struct_get_struct_type_name(const cfg_struct_type *);
bool              cfg_struct_has_item(const cfg_struct_type *, const char *);
const char *      cfg_struct_get_item(const cfg_struct_type *, const char *);
bool              cfg_struct_has_sub_struct(const cfg_struct_type *, const char *);
cfg_struct_type * cfg_struct_get_sub_struct(const cfg_struct_type *, const char *);
const char *      cfg_struct_get_sub_struct_type(const cfg_struct_type *, const char *);
int               cfg_struct_num_instances_of_sub_struct_type(const cfg_struct_type *, const char *);
stringlist_type * cfg_struct_alloc_instances_of_sub_struct_type(const cfg_struct_type *, const char *);
set_type        * cfg_struct_alloc_sub_structs_type_set(const cfg_struct_type *);
#endif
