#include <stdbool.h>
#include <stdio.h>
#include <tpgzone.h>

int main(int argc, char ** argv)
{
  if(argc != 2)
  {
    printf("Usage: tpgzone_test.x config_file\n");
    return 0;
  }

  tpgzone_type * tpgzone = tpgzone_fscanf_alloc(argv[1], true);

  tpgzone_free(tpgzone);
  return 0;
}
