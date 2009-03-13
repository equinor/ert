#include <string.h>
#include <sqlite3_driver.h>
#include <stdio.h>


static const char SQLITE3_DB_FILE[]         = "my_sqlite3_db.bin";


int main(
  int argc,
  char ** argv)
{
  const char * id             = "poro";
  int          realization_nr = 1;
  int          restart_nr     = 123;
  int          state          = 0;
  const char * data           = "svadafesalsdfhalkdfhasldkfahklsdfhasdkfjht";
  int          bytesize_data  = strlen(data) + 1;

  char       * my_data;
  int          bytesize_my_data;

  sqlite3_driver_type * driver  = sqlite3_driver_alloc(SQLITE3_DB_FILE, "default");
  void                * _driver = (void *) driver;

  printf("Attempting to save a node.\n");
  sqlite3_driver_save_node( _driver, id, realization_nr, restart_nr, state, data, bytesize_data);
  if( sqlite3_driver_has_node( _driver, id, realization_nr, restart_nr, state) )
    printf("Successfully saved a node.\n");
  else
    printf("Failed to save a node!!\n");


  printf("Attempting to load the node.\n");
  if( sqlite3_driver_load_node( _driver, id, realization_nr, restart_nr, state, (void **) &my_data, &bytesize_my_data) )
  {
    printf("Loaded: %s\n", my_data);
    free(my_data);
  }
  else
    printf("Coulnd't find the node.\n");


  printf("Deleting node.\n");
  sqlite3_driver_unlink_node( _driver, id, realization_nr, restart_nr, state);
  if( sqlite3_driver_has_node( _driver, id, realization_nr, restart_nr, state) )
    printf("Failed to delete node!!!\n");
  else
    printf("Successfully deleted node.\n");

  sqlite3_driver_free( _driver );
  return 0;
};
