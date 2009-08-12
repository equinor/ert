#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <basic_driver.h>
#include <sqlite3_driver.h>
#include <buffer.h>
#include <fs_types.h>
#include <enkf_config_node.h>



static const char SQL_CREATE_TABLE_fs[]     = "CREATE TABLE IF NOT EXISTS fs  "
  "(id TEXT NOT NULL, casename TEXT NOT NULL, realization_nr INTEGER NOT NULL,"
  "restart_nr INTEGER NOT NULL, data BLOB NOT NULL,   "
  "UNIQUE (id, casename, realization_nr, restart_nr));                 ";

static const char SQL_INSERT_INTO_fs[]      = "REPLACE INTO fs (id, casename, "
  "realization_nr, restart_nr, data) values (?, ?, ?, ?, ?);        ";

static const char SQL_SELECT_DATA_FROM_fs[] = "SELECT data FROM fs WHERE id=? "
  "AND casename=? AND realization_nr=? AND restart_nr=? ;          ";

static const char SQL_DELETE_FROM_fs[] = "DELETE FROM fs WHERE id=?           "
  "AND casename=? AND realization_nr=? AND restart_nr=? ;          ";



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
  if(driver->__id != SQLITE_DRIVER_ID)
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
  sqlite3_stmt * stmt,    /* Prepared sqlite3 SQL statement.              */
  int            index,   /* Position of '?' to replace. Starting at 1.   */
  int            value)   /* Integer value to substitute.                 */
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
  ensure that "text" is not free'd before stmt is finalized.
*/
static
void bind_text(
  sqlite3_stmt * stmt,  /* Prepared sqlite3 SQL statement.              */
  int            index, /* Position of '?' to replace. Starting at 1.   */
  const char   * text)  /* \0 terminated string to substitute.          */
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
  ensure that "data" is not free'd before stmt is finalized.
*/
static
void bind_blob(
  sqlite3_stmt * stmt,          /* Prepared sqlite3 SQL statement.            */
  int            index,         /* Position of '?' to replace. Starting at 1. */
  const void   * data,          /* Binary value to substitute.                */
  int            bytesize_data) /* Bytesize of the binary value.              */
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
static int execute_plain_sql(sqlite3    * db,  const char * sql)
{
  char * err_msg;
  int    result  = sqlite3_exec(db, sql, NULL, NULL, &err_msg);

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

  result = sqlite3_open_v2(db_file, &db,
                           SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
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
void sqlite3_driver_save_node(void                        * _driver,
                              const enkf_config_node_type * config_node , 
                              int          report_step,
                              int          iens,
                              buffer_type * buffer) {

  int result; 
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast(_driver);
  sqlite3      * db            = driver->db;
  sqlite3_stmt * stmt          = NULL;
  const char   * casename      = driver->casename; 
  
  result = sqlite3_prepare_v2(db, SQL_INSERT_INTO_fs,
                              strlen(SQL_INSERT_INTO_fs), &stmt, NULL);
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
  bind_text(stmt, 1, enkf_config_node_get_key( config_node ) );
  bind_text(stmt, 2, casename           );
  bind_int( stmt, 3, iens     );
  bind_int( stmt, 4, report_step         );
  bind_blob(stmt, 5, buffer_get_data( buffer ) , buffer_get_size( buffer ));


  result = sqlite3_step(stmt);
  if(result != SQLITE_DONE)
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n",
            result, sqlite3_errmsg(db));
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
  Returns true if the node can be found in the database.

  If data and bytesize_data are not NULL, the data field and it's
  bytesize are copied to these fields. It is the calling scope's
  responsibility to free the alloc'd data.
*/
bool sqlite3_driver_load_node(void * _driver, const enkf_config_node_type * config_node, int report_step ,  int iens , buffer_type * buffer) {
  int  result; 
  bool has_node = false;
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast(_driver);
  sqlite3      * db            = driver->db;
  sqlite3_stmt * stmt          = NULL;
  const char   * casename      = driver->casename;

  result = sqlite3_prepare_v2(db, SQL_SELECT_DATA_FROM_fs,
                              strlen(SQL_SELECT_DATA_FROM_fs), &stmt, NULL);
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
  bind_text(stmt, 1, enkf_config_node_get_key( config_node ) );
  bind_text(stmt, 2, casename );
  bind_int( stmt, 3, iens );
  bind_int( stmt, 4, report_step );
  

  /**
    Check if we have the requested node. Since there is a UNIQUE lock in
    SQL_CREATE_TABLE_fs, there can be at most one result.
  */
  result = sqlite3_step(stmt);
  if(result == SQLITE_DONE)
    has_node = false;
  else if(result == SQLITE_ROW)
    has_node = true;
  else
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n",
            result, sqlite3_errmsg(db));
    sqlite3_close(db);
    util_abort("%s: Internal error.\n", __func__);
  }
  

  /**
    Copy the result if wanted.
  */
  if(buffer != NULL && has_node) {
    const void * my_blob = sqlite3_column_blob(stmt, 0);
    int bytesize_my_blob = sqlite3_column_bytes(stmt, 0);
    buffer_fwrite( buffer , my_blob , 1 , bytesize_my_blob);
    buffer_fseek( buffer , 0L , SEEK_SET);
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
  const enkf_config_node_type * config_node,
  int          realization_nr,
  int          restart_nr)
{
  int  result; 
  sqlite3_driver_type * driver   = sqlite3_driver_safe_cast(_driver);
  sqlite3             * db       = driver->db;
  sqlite3_stmt        * stmt     = NULL;
  const char          * casename = driver->casename;


  result = sqlite3_prepare_v2(db, SQL_DELETE_FROM_fs,
                              strlen(SQL_DELETE_FROM_fs), &stmt, NULL);
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
  bind_text(stmt, 1, enkf_config_node_get_key( config_node ) );
  bind_text(stmt, 2, casename      );
  bind_int( stmt, 3, realization_nr);
  bind_int( stmt, 4, restart_nr    );


  result = sqlite3_step(stmt);
  if(result != SQLITE_DONE)
  {
    fprintf(stderr, "Internal error, giving up. Error number %d: %s\n",
            result, sqlite3_errmsg(db));
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
  const enkf_config_node_type * config_node , 
  int          realization_nr,
  int          restart_nr ) 
{
  return sqlite3_driver_load_node(_driver, config_node , realization_nr, restart_nr, NULL);
}


/**
   Change casename.
*/
void sqlite3_driver_change_casename(  void * _driver,  const char * casename, bool read) {
  sqlite3_driver_type * driver = sqlite3_driver_safe_cast( _driver );
  driver->casename = util_realloc_string_copy(driver->casename, casename);
}


/*****************************************************************/


void sqlite3_driver_fwrite_mount_info(FILE * stream , fs_driver_type driver_type , const char * db_file ) {
  util_fwrite_int(driver_type , stream);
  util_fwrite_int(SQLITE_DRIVER_ID , stream);
  util_fwrite_string(db_file , stream);
}

/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/
sqlite3_driver_type * sqlite3_driver_fread_alloc(const char * root_path , FILE * stream) {
  char * db_file = util_fread_alloc_string( stream );
  sqlite3_driver_type * driver = sqlite3_driver_alloc(root_path , db_file );
  free(db_file);
  return driver;
}


/**
  Allocate a sqlite3_driver_type.

  This function will also create the fs table in the
  db_file if it does not already exist.
*/
void * sqlite3_driver_alloc(const char * root_path , const char * db_file) {
  sqlite3_driver_type * driver = util_malloc(sizeof * driver, __func__);

  driver->__id     = SQLITE_DRIVER_ID;
  driver->db_file  = util_alloc_filename( root_path , db_file , NULL);
  driver->casename = NULL;
  driver->db       = open_db(driver->db_file);
  create_fs_table_if_not_exists(driver->db);

  
  driver->load 	      = sqlite3_driver_load_node;  /* This returns bool - whereas the funtion prototype is void. */
  driver->save 	      = sqlite3_driver_save_node;
  driver->has_node    = sqlite3_driver_has_node;
  driver->unlink_node = sqlite3_driver_unlink_node;
  driver->free_driver = sqlite3_driver_free;
  driver->select_dir  = sqlite3_driver_change_casename;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init( basic_driver );
    return basic_driver;
  }
}

