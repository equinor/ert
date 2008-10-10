#include <cfg_util.h>
#include <cfg_lang.h>
#include <string.h>

int main()
{
  char * pad_keys[] = {"{","}","="};
  char * buffer = cfg_util_alloc_token_buffer("svada.txt", "--", 3, pad_keys);

  printf("buffer : %s\n", buffer);

  cfg_type_def_type * cfg_type_def = cfg_type_def_alloc_from_buffer(&buffer, &buffer, "svada");

  cfg_type_def_printf(cfg_type_def);
  return 0;
}
