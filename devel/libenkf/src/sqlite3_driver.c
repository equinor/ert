#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <basic_driver.h>
#include <sqlite3_driver.h>

/**
  THIS ID IS TEMPORARY AND SHALL BE MOVED TO fs_types.h
*/
static int SQLITE3_DRIVER_ID = 99792683;


static const char SQL_CREATE_TABLE_fs[]     = "CREATE TABLE IF NOT EXISTS fs  "
  "(id TEXT NOT NULL, casename TEXT NOT NULL, realization_nr INTEGER NOT NULL,"
  "restart_nr INTEGER NOT NULL, state INTEGER NOT NULL, data BLOB NOT NULL,   "
  "UNIQUE (id, casename, realization_nr, restart_nr, state));                 ";
static const char SQL_INSERT_INTO_fs[]      = "REPLACE INTO fs (id, casename, "
  "realization_nr, restart_nr, state, data) values (?, ?, ?, ?, ?, ?);        ";
static const char SQL_SELECT_DATA_FROM_fs[] = "SELECT data FROM fs WHERE id=? "
  "AND casename=? AND realization_nr=? AND restart_nr=? AND state=?;          ";
static const char SQL_DELETE_FROM_fs[] = "DELETE FROM fs WHERE id=?           "
  "AND casename=? AND realization_nr=? AND restart_nr=? AND state=?;          ";



struct sqlite3_driver_struct {
  BASIC_DRIVER_FIELDS;        /* Function pointers .              */
  int       __id;             /* Used for safe casting.           */
  sqlite3 * db;               /* Connection to sqlite3 db.        */
  char    * db_file;          /* File to store the sqlite3 db in. */
  char    * casename;         /* Casename.                        */
};



static
sqlite3_driver_type * sqlite3_driver_safe_cast(
  void * _driver)
{
  sqlite3_driver_type * driver = (sqlite3_driver_type *) _driver;
  if(driver->__id != SQLITE3_DRIVER_ID)
    util_abort("%s: internal error - cast failed.\n", __func__);

  return driver;
}


/**
  Get the bytesize of a string.
*/
static
int get_bytesize(
  const char * str)
{
  return (strlen(str) + 1) * sizeof * str;
}



/**
  Binds an int to an SQL statement.
*/
static
void bind_int(
  sqlite3_stmt * stmt,
  int            index,
  int            value)
{
  int result = sqlite3_bind_int(stmt, index, value);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Failed to bind integer %d to index %d.\n", value, index);
    util_abort("%s: Internal error.\n", __func__);
  }
}



/**
  Binds a \0 terminated char * to an SQL statement.

  WARNING:

  For efficiency, this function does *NOT* take a local copy
  of "text". Thus, it is the calling scope's responsibility to
  ensure that "text" is not free'd before stmt.
*/
static
void bind_text(
  sqlite3_stmt * stmt,
  int            index,
  const char   * text)
{
  int result = sqlite3_bind_text(stmt, index, text, get_bytesize(text), SQLITE_STATIC);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Failed to bind string %s to index %d.\n", text, index);
    util_abort("%s: Internal error.\n", __func__);
  }
}



/**
  Binds binary data to an SQL statement.

  WARNING:

  For efficiency, this function does *NOT* take a local copy
  of "data". Thus, it is the calling scope's responsibility to
  ensure that "data" is not free'd before stmt.
*/
static
void bind_blob(
  sqlite3_stmt * stmt,
  int            index,
  const void   * data,
  int            bytesize_data)
{
  int result = sqlite3_bind_blob(stmt, index, data, bytesize_data, SQLITE_STATIC);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Failed to bind blob to index %d.\n", index);
    util_abort("%s: Internal error.\n", __func__);
  }
}



/**
  Execute a plain SQL string on db. Useful for creating tables etc.
*/
static
int execute_plain_sql(
  sqlite3    * db,
  const char * sql)
{
  char * err_msg;
  int result = sqlite3_exec(db, sql, NULL, NULL, &err_msg);

  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Failed to execute SQL statement: %s\n", sql);
    fprintf(stderr, "Error was: %s\n", err_msg);
    sqlite3_free(err_msg);
  }

  return result;
}



/**
  Open or create a sqlite3 database in db_file.
*/
static
sqlite3 * open_db(
  const char * db_file)
{
  int       result;
  sqlite3 * db;

  result = sqlite3_open_v2(db_file, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Error when trying to open \"%s\". Error number %d: %s\n",
            db_file, result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }

  return db;
}



/**
  This function creates the fs table if it does not already exists.

  See SQL_CREATE_TABLE_fs in top of this file for the statement.
*/
static
void create_fs_table_if_not_exists(
  sqlite3 * db)
{
  int result;

  result = execute_plain_sql(db, SQL_CREATE_TABLE_fs);
  if(result != SQLITE_OK)
  {
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }
}



/**
  Allocate a sqlite3_driver_type.

  This function will also create the fs table in the
  db_file if it does not already exist.
*/
sqlite3_driver_type * sqlite3_driver_alloc(
  const char * db_file,
  const char * casename)
{
  sqlite3_driver_type * driver = util_malloc(sizeof * driver, __func__);

  driver->__id     = SQLITE3_DRIVER_ID;
  driver->db_file  = util_alloc_string_copy(db_file);
  driver->casename = util_alloc_string_copy(casename);
  driver->db       = open_db(driver->db_file);

  create_fs_table_if_not_exists(driver->db);

  return driver;
}



/**
  Free a sqlite3_driver_type.
*/
void sqlite3_driver_free(
  void * _driver)
{
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast(_driver);
  free(driver->db_file);
  free(driver->casename);
  sqlite3_close(driver->db);
  free(driver);
}



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
  int          bytesize_data)
{
  int result; 
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast(_driver);
  sqlite3      * db            = driver->db;
  sqlite3_stmt * stmt          = NULL;
  const char   * casename      = driver->casename; 

  result = sqlite3_prepare_v2(db, SQL_INSERT_INTO_fs, strlen(SQL_INSERT_INTO_fs), &stmt, NULL);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Error preparing statement \"%s\". Error number %d: %s\n", 
            SQL_INSERT_INTO_fs, result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }

  /**
    Bind data to the '?' placeholders in the SQL_INSERT_INTO_fs statement.
  */
  bind_text(stmt, 1, id                 );
  bind_text(stmt, 2, casename           );
  bind_int( stmt, 3, realization_nr     );
  bind_int( stmt, 4, restart_nr         );
  bind_int( stmt, 5, state              );
  bind_blob(stmt, 6, data, bytesize_data);


  result = sqlite3_step(stmt);
  if(result != SQLITE_DONE)
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n", result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }

  result = sqlite3_finalize(stmt);
  if(result != SQLITE_OK )
  {
    fprintf(stderr, "Failed to finalize SQL statement. Error number %d: %s\n", result,
            sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }
}



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
  int        * bytesize_data)
{
  int  result; 
  bool has_node = false;
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast(_driver);
  sqlite3      * db            = driver->db;
  sqlite3_stmt * stmt          = NULL;
  const char   * casename      = driver->casename;

  result = sqlite3_prepare_v2(db, SQL_SELECT_DATA_FROM_fs, strlen(SQL_SELECT_DATA_FROM_fs),
                              &stmt, NULL);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Error preparing statement \"%s\". Error number %d: %s\n",
            SQL_SELECT_DATA_FROM_fs, result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }


  /**
    Bind data to the '?' placeholders in the SQL_SELECT_DATA_FROM_fs.
  */
  bind_text(stmt, 1, id            );
  bind_text(stmt, 2, casename      );
  bind_int( stmt, 3, realization_nr);
  bind_int( stmt, 4, restart_nr    );
  bind_int( stmt, 5, state         );


  /**
    Check if we have the requested node. Since there is a UNIQUE lock in SQL_CREATE_TABLE_fs,
    there can be at most one result.
  */
  result = sqlite3_step(stmt);
  if(result == SQLITE_DONE)
    has_node = false;
  else if(result == SQLITE_ROW)
    has_node = true;
  else
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n", result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }
  

  /**
    Copy the result if wanted.
  */
  if(data != NULL && bytesize_data != NULL && has_node)
  {
    const void * my_blob = sqlite3_column_blob(stmt, 0);
    int bytesize_my_blob = sqlite3_column_bytes(stmt, 0);
    *data = malloc(bytesize_my_blob);
    memcpy(*data, my_blob, bytesize_my_blob);
    *bytesize_data = bytesize_my_blob;
  }


  result = sqlite3_finalize(stmt);
  if(result != SQLITE_OK )
  {
    fprintf(stderr, "Failed to finalize SQL statement. Error number %d: %s\n",
            result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }


  return has_node;
}



/**
  Delete a node with these exact values if it exists in the fs table.
*/
void sqlite3_driver_unlink_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state)
{
  int  result; 
  sqlite3_driver_type * driver   = sqlite3_driver_safe_cast(_driver);
  sqlite3             * db       = driver->db;
  sqlite3_stmt        * stmt     = NULL;
  const char          * casename = driver->casename;


  result = sqlite3_prepare_v2(db, SQL_DELETE_FROM_fs, strlen(SQL_DELETE_FROM_fs),
                              &stmt, NULL);
  if(result != SQLITE_OK)
  {
    fprintf(stderr, "Error preparing statement \"%s\". Error number %d: %s\n",
            SQL_DELETE_FROM_fs, result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }


  /**
    Bind data to the '?' placeholders in the SQL_DELETE_FROM_fs.
  */
  bind_text(stmt, 1, id            );
  bind_text(stmt, 2, casename      );
  bind_int( stmt, 3, realization_nr);
  bind_int( stmt, 4, restart_nr    );
  bind_int( stmt, 5, state         );


  result = sqlite3_step(stmt);
  if(result != SQLITE_DONE)
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n", result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }

  result = sqlite3_finalize(stmt);
  if(result != SQLITE_OK )
  {
    fprintf(stderr, "Failed to finalize SQL statement. Error number %d: %s\n",
            result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }
}



/**
  Checks if a node with these exact values is present in the fs table.
*/
bool sqlite3_driver_has_node(
  void       * _driver,
  const char * id,
  int          realization_nr,
  int          restart_nr,
  int          state)
{
  return sqlite3_driver_load_node(_driver, id, realization_nr, restart_nr, state, NULL, NULL);
}
