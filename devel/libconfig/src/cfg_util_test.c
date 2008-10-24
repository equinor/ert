#include <cfg_util.h>
#include <cfg_struct_def.h>
#include <string.h>

int main()
{
  char * pad_keys[] = {"{","}","=",";"};
  char * buffer = cfg_util_alloc_token_buffer("svada.txt", "--", 4, pad_keys);
  char * buffer_pos = buffer;

  cfg_struct_def_type * cfg_struct_def  = cfg_struct_def_alloc_from_buffer(&buffer_pos, "svada", true);

  cfg_struct_def_free(cfg_struct_def);
  free(buffer);

  return 0;
}
