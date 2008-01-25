#include <basic_driver.h>


#define BASIC_TYPE_ID 1000

void basic_driver_init(basic_driver_type * driver) {
  driver->basic_type_id = BASIC_TYPE_ID;
}



void basic_driver_assert_cast(const basic_driver_type * driver) {
  if (driver->basic_type_id != BASIC_TYPE_ID) {
    fprintf(stderr,"%s: internal error - incorrect cast() - aborting \n" , __func__);
    abort();
  }
}


