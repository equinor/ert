#include <string.h>
#include <stdlib.h>
#include <fs_types.h>
#include <util.h>


fs_driver_impl fs_types_lookup_string_name(const char * driver_name) {
  if (strcmp(driver_name , "PLAIN") == 0)
    return PLAIN_DRIVER_ID;
  else if (strcmp(driver_name , "SQLITE") == 0)
    return SQLITE_DRIVER_ID;
  else {
    util_abort("%s: could not determine driver type for input:%s \n",__func__ , driver_name);
    return INVALID_DRIVER_ID;
  }
}
