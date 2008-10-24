#ifndef __CFG_LANG_H__
#define __CFG_LANG_H__
#include <stdbool.h>
#include <hash.h>
#include <set.h>
#include <data_type.h>


typedef struct cfg_item_def_struct   cfg_item_def_type;
typedef struct cfg_struct_def_struct cfg_struct_def_type;


cfg_struct_def_type * cfg_struct_def_alloc_from_buffer(char **, const char *, bool);
void                  cfg_struct_def_free(cfg_struct_def_type *);



/**********************************************************************************/



struct cfg_item_def_struct{
  char * name;
  data_type_enum   data_type;
  set_type       * restriction;

  char * help;
};



struct cfg_struct_def_struct{
  char * name;

  hash_type * sub_items;  
  hash_type * sub_structs;
  set_type  * required;   

  char * help;
};


#endif
