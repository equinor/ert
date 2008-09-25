#include <stdbool.h>
#include <stdio.h>
#include <util.h>
#include <tpgzone.h>

int main(int argc, char ** argv)
{
  bool endian_flip = true;

  char * config_file;
  char * path;
  char * ext;

  if(argc != 2)
  {
    printf("Usage: tpgzone.x config_file\n");
    return 0;
  }

//  util_alloc_file_components(argv[1], &path, &config_file, &ext);
//
//  if(path != NULL)
//  {
//    /*
//      If path is an absolute path, we are running in batch with the same config.
//    */
//    if(!util_is_abs_path(path))
//      chdir(path);
//  }

  config_file = util_strcat_realloc(config_file,".");
  config_file = util_strcat_realloc(config_file,ext);
  
  tpgzone_type * tpgzone = tpgzone_fscanf_alloc(config_file, endian_flip);

  tpgzone_apply(tpgzone, endian_flip);

  tpgzone_free(tpgzone);

  free(config_file);
  free(path);
  free(ext);

  return 0;
}
