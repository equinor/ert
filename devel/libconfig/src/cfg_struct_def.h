#ifndef __CFG_STRUCT_DEF_H__
#define __CFG_STRUCT_DEF_H__
#include <stdbool.h>
#include <hash.h>
#include <set.h>
#include <data_type.h>


typedef struct cfg_item_def_struct   cfg_item_def_type;
typedef struct cfg_struct_def_struct cfg_struct_def_type;


cfg_struct_def_type * cfg_struct_def_fscanf_alloc(const char *);
void                  cfg_struct_def_free(cfg_struct_def_type *);

void                  cfg_item_def_printf_help(cfg_item_def_type *);



/**********************************************************************************/



struct cfg_item_def_struct{
  char * name;
  data_type_enum   data_type;
  set_type       * restriction;

  char * default_value;
  char * help;
};



struct cfg_struct_def_struct{
  char * name;

  hash_type * sub_items;  
  hash_type * sub_structs;
  set_type  * required_sub_items;
  set_type  * required_sub_structs;

  char * help;
};


#endif
