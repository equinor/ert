#include <util.h>
#include <fs_types.h>
#include <basic_driver.h>




void basic_driver_init(basic_driver_type * driver) {
  driver->type_id = BASIC_DRIVER_ID;
}

void basic_driver_assert_cast(const basic_driver_type * driver) {
  if (driver->type_id != BASIC_DRIVER_ID) 
    util_abort("%s: internal error - incorrect cast() - aborting \n" , __func__);
}


basic_driver_type * basic_driver_safe_cast(void * __driver) {
  basic_driver_type * driver = (basic_driver_type *) __driver;
  if (driver->type_id != BASIC_DRIVER_ID)
    util_abort("%s: runtime cast failed. \n",__func__);
  return driver;
}

/*****************************************************************/

void basic_static_driver_init(basic_static_driver_type * driver) {
  driver->type_id = BASIC_DRIVER_ID;
}

void basic_static_driver_assert_cast(const basic_static_driver_type * driver) {
  if (driver->type_id != BASIC_DRIVER_ID) 
    util_abort("%s: internal error - incorrect cast() - aborting \n" , __func__);
}


basic_static_driver_type * basic_static_driver_safe_cast(void * __driver) {
  basic_static_driver_type * driver = (basic_static_driver_type *) __driver;
  if (driver->type_id != BASIC_DRIVER_ID)
    util_abort("%s: runtime cast failed. \n",__func__);
  return driver;
}


/*****************************************************************/


void basic_obs_driver_init(basic_obs_driver_type * driver) {
  driver->type_id = BASIC_DRIVER_ID;
}

void basic_obs_driver_assert_cast(const basic_obs_driver_type * driver) {
  if (driver->type_id != BASIC_DRIVER_ID) 
    util_abort("%s: internal error - incorrect cast() - aborting \n" , __func__);
}


basic_obs_driver_type * basic_obs_driver_safe_cast(void * __driver) {
  basic_obs_driver_type * driver = (basic_obs_driver_type *) __driver;
  if (driver->type_id != BASIC_DRIVER_ID)
    util_abort("%s: runtime cast failed. \n",__func__);
  return driver;
}

