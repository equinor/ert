#include <hash.h>
#include <util.h>
#include <gruptree.h>

int main()
{
  gruptree_type * gruptree = gruptree_alloc_emtpy();
  gruptree_register_grup(gruptree, "INJE", "FIELD");
  gruptree_register_well(gruptree, "WI1"  , "INJE");
  gruptree_register_well(gruptree, "WI2"  , "INJE");
  gruptree_register_well(gruptree, "WI3"  , "INJE");
  gruptree_register_well(gruptree, "WI4"  , "INJE");

  gruptree_register_grup(gruptree, "PLATA", "FIELD");
  gruptree_register_grup(gruptree, "INJE", "PLATA");
  gruptree_register_grup(gruptree, "PROD", "PLATA");
  gruptree_register_well(gruptree, "WP1",  "PROD");
  gruptree_register_well(gruptree, "WP2",  "PROD");
  gruptree_register_well(gruptree, "WP3",  "PROD");
  gruptree_register_well(gruptree, "WP4",  "PROD");
  gruptree_register_well(gruptree, "WP5",  "PROD");

  gruptree_printf_wells(gruptree, "FIELD");
  gruptree_printf_wells(gruptree, "PLATA");
  gruptree_printf_wells(gruptree, "INJE");
  gruptree_printf_wells(gruptree, "PROD");


  gruptree_free(gruptree);
  return 0;
}
