#include <cfg_util.h>
#include <cfg_struct_def.h>
#include <cfg_struct.h>
#include <string.h>

int main()
{

  cfg_struct_def_type * cfg_struct_def = cfg_struct_def_alloc_from_file("def.txt"); 
  
  cfg_struct_type * cfg_struct = cfg_struct_alloc_from_file("cfg.txt", cfg_struct_def);
  cfg_struct_free(cfg_struct);



  cfg_struct_def_free(cfg_struct_def);

  return 0;
}
