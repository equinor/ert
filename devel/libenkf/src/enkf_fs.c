#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <util.h>
#include <enkf_fs.h>
#include <path_fmt.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <fs_index.h>
#include <fs_types.h>
#include <ecl_static_kw.h>
#include <plain_driver_parameter.h>
#include <plain_driver_static.h>
#include <plain_driver_dynamic.h>


/**

  About storage in the EnKF system
  ================================

  The system for storage in the EnKF system is quite complicated, maybe too
  complicated. The reason the system is so complex is (at least) twofold:

    1. It is a goal that it should be relatively easy to write new systems
       (i.e. drivers) for storage. The current suite of drivers
       (plain_driver_xx) are based on normal (compressed) fread() and fwrite()
       system calls. But with the current implementation one could write e.g. a
       MySQL based driver for storage without touching(??) the rest of the EnKF
       code.

    2. The parametrs / static restart data / dynamic restart data have very
       different storage characteristics. By splitting the storage up in three
       different drivers we can write drivers which are specialized for the
       different types of data.


  

  The interface
  -------------     

  The unit of storage in the enkf_fs system is one enkf_node instance. The
  interface between the storage system and the rest of the EnKF system is
  through the enkf_fs functions:

    enkf_fs_fread_node()
    enkf_fs_has_node()
    enkf_fs_fwrite_node()
    enkf_fs_unlink_node() 

  So all these functions (partly except enkf_fs_has_node()) work on a enkf_node
  instance, and in addition they take the following input:

    - iens        : ensemble member number
    - report_step : the report_step number we are interested in
    - state       : whether we are considering an analyzed node or a forecast.  

  In addition to the functions enkf_fs_fread_node() and enkf_fs_fwrite_node() there
  are higher level functions enkf_fs_fread_alloc_ensemble() to load an ensemble of
  nodes and enkf_fs_fread_alloc_ts() to load a time series of nodes. The driver can
  implement these functions (to get a performance gain when loading many nodes). It
  is not necceasry to implement these functions, the enkf_fs layer has simple
  functions doing repeated calls to the enkf_fs_fread_node() function.


  The drivers
  -----------

  The enkf_fs layer does not self implement the functions to read and write
  nodes. Instead what happens is:

    1. We determine the type of the node (static/dynamic/parameter), and select
       the appropriate driver.

    2. The appropriate driver is called to implement e.g. the fread_node
       functions.

  The different types of data have different characteristcs, which the driver is
  implemented to support. The characteristics the drivers support are the
  following:


  dynamic driver
  --------------
  This is the simplest driver, all data is stored both in a forecast version and
  an analyzed version.


  parameter driver
  ----------------
  This driver utilizes that parameters do not change during the forward model,
  i.e. (analyzed , t) = (forecast , t - 1). So, only one version of the data is
  actually stored; if you ask for the forecast you just get the data from the
  previous report_step. 
  To support spin-ups and such the driver will actually go backwards in
  report_time all the way until a node is found on disk.


  static driver
  -------------
  Like the parameter driver this also only stores one version od the data,
  however in addition it has to query the node for a ID to support multiply
  occuring keywords in ECLIPSE restart files.

  Currently only the plain_driver_xxx family has been implemented. Observe that
  there is no dependencies between the drivers, it is perfectly possible to
  implement a new driver for storage of static datat only. (There is probably a
  large amount of static data which is common both between members and for
  several consecutive report steps; utilizing that one could write a static
  driver which was admittedly slower, but leaner on the storage.)

  The drivers are allocated prior to allocating the enkf_fs instance, and
  pointers are passed in when allocating the enkf_fs instance.


  Mounting the filesystem
  -----------------------

  Mounting the filesystem - cool ehh?? Anyway, the important point is
  that the moment ensemble information has hit the filesystem later
  versions of the enkf program must support exactly that lay-out,
  those drivers+++. To ensure this I see two possibilities:

    1. We can freeze the filesystem drivers, and the layout on disk
       indefinetly.

    2. We can store the information needed to bootstrap the drivers,
       according to the current layout on disk, in the
       filesystem. I.e. something like a '/etc/fstab' file.

  We have chosen the second alternative. Currently this implemented as
  follows:

    1. In main() we query for the file {root-path}/enkf_mount_info. If
       that file does not exists it is created by calls to the
       selected drivers xxxx_fwrite_mount_info() functions.

    2. enkf_fs_mount() is called with the enkf_mount_info as input.

  The enkf_mount_info file (BINARY) consists of four records (one for
  each driver, including the index). The format of each record is:

     DRIVER_CATEGORY   DRIVER_ID    INFO 
     int               int          void *

  The driver category should be one of the four integer values in
  fs_driver_type (fs_types.h) and DRIVER_ID is one the integer
  values in fs_driver_impl. The last void * data is whatever
  (serialized) info the driver needs to bootstrap. This info is
  written by the drivers xxxx_fwrite_mount_info() function, and it is
  used when the driver is allocated with xxxx_fread_alloc().

  The different drivers can be in arbitrary order in the
  enkf_mount_info file, but when foru records are read it checks that
  all drivers have been initialized, and aborts if that is not the
  case.
  
  If the enkf_mount_info file is deleted you (might) be fucked. It
  is currently 'protected' with chomd a-w - but that is of course not
  foolprof.
*/




/**
   Observe the following convention: the initial ensemble at report
   step 0 is supposed to be analyzed. If we ask for the forecast at
   report_step 0, we should get the analyzed value.
*/



struct enkf_fs_struct {
  basic_driver_type         * dynamic;               /* Implements functions for read/write of dynamic data. */
  basic_static_driver_type  * eclipse_static;        /* Implements functions for read/write of static elements in ECLIPSE restart files. */
  basic_driver_type  	    * parameter;             /* Implements functions for read/write of parameters. */
  fs_index_type      	    * index;                 /* Currently only used to write restart_kw_list instances (not properly virtualized). */
  bool                        read_only;             /* Whether this filesystem has been mounted read-only. */
  int                         lock_fd;               /* Integer containing a file descriptor to lockfile. */
  char                      * lockfile; 
};







enkf_fs_type * enkf_fs_mount(const char * root_path , const char *mount_info , const char * lock_path) {
  const int num_drivers = 4;
  enkf_fs_type * fs  = util_malloc(sizeof * fs , __func__);
  fs->index          = NULL;
  fs->dynamic        = NULL;
  fs->eclipse_static = NULL;
  fs->parameter      = NULL;
  {
    char * config_file = util_alloc_full_path(root_path , mount_info);
    FILE * stream      = util_fopen(config_file , "r");
    int i;
    
    for (i=0; i < num_drivers; i++) {
      void * driver   = NULL;
      fs_driver_type driver_category = util_fread_int( stream );
      fs_driver_impl driver_id       = util_fread_int( stream );
      switch(driver_id) {
      case(PLAIN_DRIVER_INDEX_ID):
	driver = fs_index_fread_alloc( root_path , stream );
	break;
      case(PLAIN_DRIVER_STATIC_ID):
	driver = plain_driver_static_fread_alloc( root_path , stream );
	break;
      case(PLAIN_DRIVER_DYNAMIC_ID):
	driver = plain_driver_dynamic_fread_alloc( root_path , stream );
	break;
      case(PLAIN_DRIVER_PARAMETER_ID):
	driver = plain_driver_parameter_fread_alloc( root_path , stream );
	break;
      default:
	util_abort("%s: fatal error in mount_map:%s - driver ID:%d not recognized \n",__func__ , config_file , driver_id);
      }
	

      switch(driver_category) {
      case(PARAMETER_DRIVER):
	fs->parameter = driver;
	break;
      case(STATIC_DRIVER):
	fs->eclipse_static = driver;
	break;
      case(DYNAMIC_DRIVER):
	fs->dynamic = driver;
	break;
      case(INDEX_DRIVER):
	fs->index = driver;
	break;
      default:
	util_abort("%s: fatal error in mount_map:%s - driver category:%d not recognized \n",__func__ , config_file , driver_category);
      }
    }

    if (fs->parameter  	   == NULL) util_abort("%s: fatal error - mount map in:%s did not contain parameter driver.\n", __func__ , config_file);
    if (fs->dynamic    	   == NULL) util_abort("%s: fatal error - mount map in:%s did not contain dynamic driver.\n",   __func__ , config_file);
    if (fs->eclipse_static == NULL) util_abort("%s: fatal error - mount map in:%s did not contain ecl_static driver.\n",__func__ , config_file);
    if (fs->index          == NULL) util_abort("%s: fatal error - mount map in:%s did not contain index driver.\n",     __func__ , config_file);
    fclose(stream);
    free(config_file);
    
  }
  basic_driver_assert_cast(fs->dynamic);
  basic_static_driver_assert_cast(fs->eclipse_static);
  basic_driver_assert_cast(fs->parameter);

  {
    char * ens_path    = util_alloc_string_copy(root_path);
    util_string_tr( ens_path , UTIL_PATH_SEP_CHAR , '_' );
    fs->lockfile = util_alloc_filename(lock_path , ens_path , "ensemble_lock");
    free(ens_path);
  }
  
  if (util_try_lockf(fs->lockfile , S_IWUSR + S_IWGRP, &fs->lock_fd))
    fs->read_only = false;
  else {
    fs->read_only = true;
    fprintf(stderr,"------------------------------------------------------------------------\n");
    fprintf(stderr,"| Warning: another EnKF instance has currently locked the ensemble at\n| \'%s\' for writing - this instance will be read-only.\n",root_path);
    fprintf(stderr,"-------------------------------------------------------------------------\n");
  }
  
  
  return fs;
}




static void enkf_fs_free_driver(basic_driver_type * driver) {
  driver->free_driver(driver);
}

static void enkf_fs_free_static_driver(basic_static_driver_type * driver) {
  driver->free_driver(driver);
}



void enkf_fs_free(enkf_fs_type * fs) {
  enkf_fs_free_driver(fs->dynamic);
  enkf_fs_free_driver(fs->parameter);
  enkf_fs_free_static_driver(fs->eclipse_static);
  fs_index_free(fs->index);
  close(fs->lock_fd);
  util_unlink_existing(fs->lockfile);
  free(fs->lockfile);
  free(fs);
}



static void * enkf_fs_select_driver(enkf_fs_type * fs , enkf_var_type var_type, const char * key) {
  void * driver = NULL;
  switch (var_type) {
  case(parameter):
    driver = fs->parameter;
    break;
  case(dynamic_result):
    driver = fs->dynamic;
    break;
  case(dynamic_state):
    driver = fs->dynamic;
    break;
  case(static_state):
    driver = fs->eclipse_static;
    break;
  default:
    util_abort("%s: fatal internal error - could not determine enkf_fs driver for object:%s - aborting.\n",__func__, key);
  }
  return driver;
}


static int enkf_fs_get_static_counter(const enkf_node_type * node) {
  ecl_static_kw_type * ecl_static = enkf_node_value_ptr( node );
  ecl_static_kw_assert_type(ecl_static);
  return ecl_static_kw_get_counter(ecl_static);
}


/*****************************************************************/
/* Exported functions for enkf_node instances . */

void enkf_fs_fwrite_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  if (enkf_fs->read_only)
    util_abort("%s: attempt to write to read_only filesystem - aborting. \n",__func__);
  {
    enkf_var_type var_type = enkf_node_get_var_type(enkf_node);
    void * _driver = enkf_fs_select_driver(enkf_fs , var_type , enkf_node_get_key(enkf_node));
    if (var_type == static_state) {
      basic_static_driver_type * driver = basic_static_driver_safe_cast(_driver);
      int static_counter = enkf_fs_get_static_counter(enkf_node);
      driver->save(driver , report_step , iens , state , static_counter , enkf_node); 
    } else {
      basic_driver_type * driver = basic_driver_safe_cast(_driver);
      
      if (report_step == 0) {
	enkf_impl_type impl_type = enkf_node_get_impl_type(enkf_node);
	if (impl_type == SUMMARY) return;    /* For report step == 0 the summary data is just garbage. */
      }
      driver->save(driver , report_step , iens , state , enkf_node); 
    }
  }
}


void enkf_fs_fread_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  enkf_var_type var_type = enkf_node_get_var_type(enkf_node);
  void * _driver = enkf_fs_select_driver(enkf_fs , var_type , enkf_node_get_key(enkf_node));
  if (var_type == static_state) {
    basic_static_driver_type * driver = basic_static_driver_safe_cast(_driver);
    int static_counter = enkf_fs_get_static_counter(enkf_node);
    driver->load(driver , report_step , iens , state , static_counter , enkf_node); 
  } else {
    basic_driver_type * driver = basic_driver_safe_cast(_driver);
    driver->load(driver , report_step , iens , state , enkf_node); 
  }
}


bool enkf_fs_has_node(enkf_fs_type * enkf_fs , const enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  {
    const char * key = enkf_config_node_get_key_ref(config_node);
    if (var_type == static_state) {
      basic_static_driver_type * driver = basic_static_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , key));
      int static_counter = 0; /* This one is impossible to get correctly hold of ... the driver aborts.*/
      return driver->has_node(driver , report_step , iens , state , static_counter , key); 
    } else {
      basic_driver_type * driver = basic_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , key));
      return driver->has_node(driver , report_step , iens , state , key); 
    }
  }
}

enkf_node_type * enkf_fs_fread_alloc_node(enkf_fs_type * enkf_fs , enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  enkf_node_type * node = enkf_node_alloc(config_node);
  enkf_fs_fread_node(enkf_fs , node , report_step , iens , state);
  return node;
}

/*****************************************************************/
/* High level functions to work on an ensemble or a time-series. */
/* Observe that both for the time_based functions, and the ensemble
   based functions both limits are INCLUSIVE. */
   

enkf_node_type ** enkf_fs_fread_alloc_ensemble(enkf_fs_type * enkf_fs , enkf_config_node_type * config_node , int report_step , int iens1 , int iens2 , state_enum state) {
  int iens;
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  if (var_type == static_state) {
    /* Static drivers *NEVER* have high level functions fall back to simple loop immediately. */
    enkf_node_type ** node_list = util_malloc( (iens2 - iens1  + 1) * sizeof * node_list , __func__);

    for (iens = iens1; iens <= iens2; iens++) {
      if (enkf_fs_has_node(enkf_fs , config_node , report_step , iens , state))
	node_list[iens - iens1] = enkf_fs_fread_alloc_node(enkf_fs , config_node , report_step , iens , state);
      else
	node_list[iens - iens1] = NULL; /* This report_step/iens does not exist in the filesystem. */ 
    }
    return node_list;

  } else {
    const char * key = enkf_config_node_get_key_ref(config_node);
    basic_driver_type * driver = basic_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , key));
    if (driver->load_ensemble != NULL)
      return driver->load_ensemble(driver , report_step , iens1 , iens2 , state , config_node);
    else {
      /* Iterative fallback */
      enkf_node_type ** node_list = util_malloc( (iens2 - iens1  + 1) * sizeof * node_list , __func__);
      for (iens = iens1; iens < iens2; iens++) {
	if (driver->has_node(driver , report_step , iens , state , key)) {
	  node_list[iens - iens1] = enkf_node_alloc(config_node);
	  driver->load( driver , report_step , iens , state , node_list[iens - iens1]);
	} else
	  node_list[iens - iens1] = NULL; /* The node does not exist in the filesystem. */
      }
      return node_list;
    }
  }
}



/**
   Observe that this function accepts state_enum == both, in that case it
   will load interleaved forecast, analyzed, forecast, analyzed , ....
*/

enkf_node_type ** enkf_fs_fread_alloc_ts(enkf_fs_type * enkf_fs , enkf_config_node_type * config_node , int step1 , int step2 ,  int iens , state_enum state) {
  int step;
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  if (var_type == static_state) {
    /* Static drivers *NEVER* have high level functions fall back to simple loop immediately. */
    enkf_node_type ** node_list = util_malloc( (step2 - step1  + 1) * sizeof * node_list , __func__);

    for (step = step1; iens <= step2; step++) {
      if (enkf_fs_has_node(enkf_fs , config_node , step , iens , state))
	node_list[step - step1] = enkf_fs_fread_alloc_node(enkf_fs , config_node , step , iens , state);
      else
	node_list[step - step1] = NULL; /* This report_step/iens does not exist in the filesystem. */ 
    }
    return node_list;
    
  } else {
    const char * key = enkf_config_node_get_key_ref(config_node);
    basic_driver_type * driver = basic_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , key));
    if (driver->load_ts != NULL)
      return driver->load_ts(driver , step1 , step2  , iens , state , config_node);
    else {
      /* Iterative fallback */

      enkf_node_type ** node_list;
      int node_index = 0;

      if (state == both)
	node_list = util_malloc( 2 * (step2 - step1  + 1) * sizeof * node_list , __func__);
      else
	node_list = util_malloc(     (step2 - step1  + 1) * sizeof * node_list , __func__);
      
      
      for (step = step1; step < step2; step++) {
	/* Loading forecast */
	if (state & forecast) {
	  if (driver->has_node(driver , step , iens , forecast , key)) {
	    node_list[node_index] = enkf_node_alloc(config_node);
	    driver->load( driver , step , iens , forecast , node_list[node_index]);
	  } else
	    node_list[node_index] = NULL; /* The node does not exist in the filesystem. */
	  node_index++;
	}
	
	/* Loading analyzed */
	if (state & analyzed) {
	  if (driver->has_node(driver , step , iens , analyzed , key)) {
	    node_list[node_index] = enkf_node_alloc(config_node);
	    driver->load( driver , step , iens , analyzed , node_list[node_index]);
	  } else
	    node_list[node_index] = NULL; /* The node does not exist in the filesystem. */
	  node_index++;
	}
      }
      return node_list;
    }
  }
}



/*****************************************************************/
/* Index related functions  . */

void enkf_fs_add_index_node(enkf_fs_type * enkf_fs , int report_step , int iens , const char * kw , enkf_var_type var_type , enkf_impl_type impl_type) {
  fs_index_add_node(enkf_fs->index , report_step , iens , kw , var_type , impl_type);
}


void enkf_fs_fwrite_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, restart_kw_list_type * kw_list) {
  fs_index_fwrite_restart_kw_list(enkf_fs->index , report_step , iens , kw_list);
}


void enkf_fs_fread_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, restart_kw_list_type * kw_list) {
  fs_index_fread_restart_kw_list(enkf_fs->index , report_step , iens , kw_list);
}



