#include <stdbool.h>
#include <stdio.h>
#include <util.h>
#include <tpgzone.h>

int main(int argc, char ** argv)
{
  char * config_file;
  char * config_base;
  char * path;
  char * ext;

  if(argc != 2)
  {
    printf("Usage: tpgzone_test.x config_file\n");
    return 0;
  }

  util_alloc_file_components(argv[1], &path, &config_base, &ext);

  if(path != NULL)
    chdir(path);

  config_file = util_strcat_realloc(config_base,".");
  config_file = util_strcat_realloc(config_file,ext);
  
  tpgzone_type * tpgzone = tpgzone_fscanf_alloc(config_file, true);

  tpgzone_free(tpgzone);
  return 0;
}
