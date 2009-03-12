#ifndef __SQLITE_DRIVER_H__
#define __SQLITE_DRIVER_H__
#include <stdbool.h>


typedef struct sqlite3_driver_struct sqlite3_driver_type;

/**
  Allocate a sqlite3_driver_type.

  This function will also create the fs table in the
  db_file if it does not already exist.
*/
sqlite3_driver_type * sqlite3_driver_alloc(
  const char * db_file,
  const char * casename);


/**
  Free a sqlite3_driver_type.
*/
void sqlite3_driver_free(
  void * _driver);


/**
  This function inserts to the fs table.

  If a node with identical id, casename, realization_nr, restart_nr and state
  exists in the same table, it is overwritten.
*/
void sqlite3_driver_save_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state,
  const void * data,
  int          bytesize_data);


/**
  Returns true if the node can be found in the database.

  If data and bytesize_data are not NULL, the data field and it's
  bytesize are copied to these fields. It is the calling scope's
  responsibility to free the alloc'd data.
*/
bool sqlite3_driver_load_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state,
  void      ** data,
  int        * bytesize_data);


/**
  Delete a node with these exact values if it exists in the fs table.
*/
void sqlite3_driver_unlink_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state);


/**
  Checks if a node with these exact values is present in the fs table.
*/
bool sqlite3_driver_has_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state);

#endif
