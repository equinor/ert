#ifndef __CFG_LANG_H__
#define __CFG_LANG_H__
#include <hash.h>
#include <set.h>

typedef struct cfg_key_def_struct cfg_key_def_type;
typedef struct cfg_type_def_struct cfg_type_def_type;

cfg_key_def_type  * cfg_key_def_alloc_from_buffer(char **, char **, const char *);
cfg_type_def_type * cfg_type_def_alloc_from_buffer(char ** , char **, const char * );

void cfg_key_def_printf(const cfg_key_def_type *);
void cfg_type_def_printf(const cfg_type_def_type *);


bool has_subtype(cfg_type_def_type *, const char *);
bool has_subkey(cfg_type_def_type *, const char *);


/**************************************************************/

/*************************************************************/
typedef enum {
              DATA_TYPE_STR,
              DATA_TYPE_INT,
              DATA_TYPE_POSINT,
              DATA_TYPE_FLOAT,
              DATA_TYPE_POSFLOAT,
              DATA_TYPE_FILE,
              DATA_TYPE_DATE
              } data_type_enum;
              


struct cfg_key_def_struct{
  char * name;
  data_type_enum   data_type;
  set_type       * restriction_set;

  char * help_text;
};


struct cfg_type_def_struct{
  char * name;

  hash_type  * sub_keys;  /* cfg_key_def_struct's. */
  hash_type  * sub_types; /* cfg_type_def_struct's. */

  set_type  * required;   /* A list of key's and types that are required. */

  char * help_text;
};

#endif
