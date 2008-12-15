#ifndef __BASIC_DRIVER_H__
#define __BASIC_DRIVER_H__ 
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_config_node.h>
#include <enkf_node.h>

typedef struct basic_driver_struct        basic_driver_type;
typedef struct basic_static_driver_struct basic_static_driver_type;

typedef void (load_node_ftype) 	  (void * , int , int , state_enum , enkf_node_type *);
typedef void (save_node_ftype) 	  (void * , int , int , state_enum , enkf_node_type *);
typedef bool (has_node_ftype)     (void * , int , int , state_enum , const char *);
typedef void (unlink_node_ftype)  (void * , int , int , state_enum , enkf_node_type *); 
typedef void (free_driver_ftype)  (void *);

typedef enkf_node_type ** (load_alloc_ensemble_ftype)    (void * , int , int , int , state_enum , enkf_config_node_type *);
typedef enkf_node_type ** (load_alloc_ts_ftype)          (void * , int , int , int , state_enum , enkf_config_node_type *);
typedef void              (save_ensemble_ftype)          (void * , int , int , int , state_enum , enkf_node_type **);
typedef void              (save_ts_ftype)                (void * , int , int , int , state_enum , enkf_node_type **);

typedef void (static_load_node_ftype) 	 (void * , int , int , state_enum , int , enkf_node_type *);
typedef void (static_save_node_ftype) 	 (void * , int , int , state_enum , int , enkf_node_type *);
typedef bool (static_has_node_ftype)     (void * , int , int , state_enum , int , const char *);
typedef void (static_unlink_node_ftype)  (void * , int , int , state_enum , int , enkf_node_type *); 


/**
   The basic_driver_type contains a number of function pointers
   and a type_id used for run-time cast checking.

   The basic_driver_type is never actually used, but the point is
   that all drivers must implement the basic driver "interface". In
   practice this is done by including the macro BASIC_DRIVER_FIELDS
   *at the start* of the definition of another driver, i.e. the simplest
   actually working driver, the plain_driver is implemented like this:

   struct plain_driver_struct {
      BASIC_DRIVER_TYPE
      int plain_driver_id;
      path_fmt_type * path;
   }


*/

   

#define BASIC_DRIVER_FIELDS   	           \
load_node_ftype    	  * load;    	   \
save_node_ftype    	  * save;    	   \
has_node_ftype     	  * has_node;      \
unlink_node_ftype  	  * unlink_node;   \
free_driver_ftype  	  * free_driver;   \
load_alloc_ensemble_ftype * load_ensemble; \
load_alloc_ts_ftype 	  * load_ts;       \
save_ensemble_ftype 	  * save_ensemble; \
save_ts_ftype       	  * save_ts;       \
int                  type_id


struct basic_driver_struct {
  BASIC_DRIVER_FIELDS;
  
};

/*****************************************************************/
/* 
   The static driver complication is because the FU***ING ECLIPSE restart
   files do not contain unique keywords, and we have to add an extra
   index (as an afterthought - yes).
*/

#define BASIC_STATIC_DRIVER_FIELDS   	 \
static_load_node_ftype    * load;    	 \
static_save_node_ftype    * save;    	 \
static_has_node_ftype     * has_node;    \
static_unlink_node_ftype  * unlink_node; \
free_driver_ftype         * free_driver; \
void * load_ensemble;\
void * load_ts;      \
void * save_ts;      \
void * save_ensemble;\
int                  type_id


struct basic_static_driver_struct { 
  BASIC_STATIC_DRIVER_FIELDS; 

}; 


/*****************************************************************/



void  	 	    basic_driver_init(basic_driver_type * );
void 	 	    basic_driver_assert_cast(const basic_driver_type * );
basic_driver_type * basic_driver_safe_cast(void * );

void 	 		   basic_static_driver_init(basic_static_driver_type * );
void 	 		   basic_static_driver_assert_cast(const basic_static_driver_type * );
basic_static_driver_type * basic_static_driver_safe_cast(void * );

#ifdef __cplusplus
}
#endif
#endif
