#ifndef __CFG_STRUCT_H__
#define __CFG_STRUCT_H__

#include <cfg_struct_def.h>

typedef struct cfg_struct_struct cfg_struct_type;

cfg_struct_type * cfg_struct_alloc_from_file(const char *, const cfg_struct_def_type *);
void cfg_struct_free(cfg_struct_type *);

#endif
