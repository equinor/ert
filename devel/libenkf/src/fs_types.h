#ifndef __FS_TYPES_H__
#define __FS_TYPES_H__



/*
  The various driver implementations - this goes on disk all over the
  place, and the numbers should be considered SET IN STONE. When a new
  driver is added the switch statement in the enkf_fs_mount() function
  must be updated.
*/
typedef enum {
  INVALID_DRIVER_ID          = 0,
  PLAIN_DRIVER_INDEX_ID      = 1001,
  PLAIN_DRIVER_STATIC_ID     = 1002,  /* Depreceated */
  PLAIN_DRIVER_DYNAMIC_ID    = 1003,  /* Depreceated */
  PLAIN_DRIVER_PARAMETER_ID  = 1004,  /* Depreceated */
  PLAIN_DRIVER_ID            = 1005,
  SQLITE_DRIVER_ID           = 2001} fs_driver_impl;







/*
  The categories of drivers. To reduce the risk of programming
  error (or at least to detect it ...), there should be no overlap
  between these ID's and the ID's of the actual implementations
  above. The same comment about permanent storage applies to these
  numbers as well.
*/

typedef enum {
  DRIVER_PARAMETER  	  = 1,
  DRIVER_STATIC     	  = 2,
  DRIVER_DYNAMIC    	  = 3, /* Depreceated */
  DRIVER_INDEX      	  = 4,  
  DRIVER_DYNAMIC_FORECAST = 5,
  DRIVER_DYNAMIC_ANALYZED = 6} fs_driver_type;



fs_driver_impl   fs_types_lookup_string_name(const char * driver_name);


#endif
