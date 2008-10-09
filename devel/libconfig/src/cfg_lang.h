#ifndef __CFG_LANG_H__
#define __CFG_LANG_H__

typedef struct cfg_key_def_struct cfg_key_def_type;
typedef struct cfg_type_def_struct cfg_type_def_type;

cfg_key_def_type  * cfg_key_def_alloc_from_tokens(int , const char ** , const char * );
cfg_type_def_type * cfg_type_def_alloc_from_tokens(int, const char ** , const char * );

void cfg_key_def_printf(const cfg_key_def_type *);
void cfg_type_def_printf(const cfg_type_def_type *);

#endif
