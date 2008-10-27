#include <cfg_util.h>
#include <cfg_struct_def.h>
#include <cfg_struct.h>
#include <string.h>
#include <stringlist.h>

int main()
{

  cfg_struct_def_type * cfg_struct_def = cfg_struct_def_fscanf_alloc("example/def.txt"); 
  cfg_struct_type * cfg_struct = cfg_struct_alloc_from_file("example/cfg.txt", cfg_struct_def);

  {
    stringlist_type * instances = cfg_struct_get_instances_of_sub_struct_type(cfg_struct, "SUMMARY_OBS");
    int num_occurences = stringlist_get_size(instances);
    for(int i=0; i<num_occurences; i++)
      printf("%i : %s\n",i, stringlist_iget(instances, i));
    stringlist_free(instances);
  }








  cfg_struct_free(cfg_struct);
  cfg_struct_def_free(cfg_struct_def);

  return 0;
}
