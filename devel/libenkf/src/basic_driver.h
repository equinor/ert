#ifndef __BASIC_DRIVER_H__
#define __BASIC_DRIVER_H__ 
#ifdef __cplusplus
extern "C" {
#endif
#include <buffer.h>
#include <stringlist.h>
#include <enkf_node.h>
#include <enkf_config_node.h>


typedef struct basic_driver_struct         basic_driver_type;
typedef struct basic_index_driver_struct   basic_driver_index_type;


typedef void (save_kwlist_ftype)  (void * , int , int , buffer_type * buffer);  /* Functions used to load/store restart_kw_list instances. */
typedef void (load_kwlist_ftype)  (void * , int , int , buffer_type * buffer);          

typedef void (select_dir_ftype)   (void * driver, const char * dir);
typedef void (load_node_ftype) 	  (void * driver, const enkf_config_node_type * , int , int , buffer_type * );
typedef void (save_node_ftype) 	  (void * driver, const enkf_config_node_type * , int , int , buffer_type * );
typedef void (unlink_node_ftype)  (void * driver, const enkf_config_node_type * , int , int );
typedef bool (has_node_ftype)     (void * driver, const enkf_config_node_type * , int , int );
typedef void (free_driver_ftype)  (void * driver);



/**
   The basic_driver_type contains a number of function pointers
   and a type_id used for run-time cast checking.
   
   The basic_driver_type is never actually used, but the point is that
   all drivers must implement the basic driver "interface". In
   practice this is done by including the macro BASIC_DRIVER_FIELDS
   *at the start* of the definition of another driver, i.e. the
   simplest (and only ...)  actually working driver, the plain_driver
   is implemented like this:

   struct plain_driver_struct {
      BASIC_DRIVER_TYPE
      int plain_driver_id;
      path_fmt_type * path;
   }


*/

   

#define BASIC_DRIVER_FIELDS   	           \
select_dir_ftype          * select_dir;    \
load_node_ftype    	  * load;    	   \
save_node_ftype    	  * save;    	   \
has_node_ftype     	  * has_node;      \
unlink_node_ftype  	  * unlink_node;   \
free_driver_ftype  	  * free_driver;   \
int                         type_id



struct basic_driver_struct {
  BASIC_DRIVER_FIELDS;
  /* Fill in whatever here - i.e. dbase state. */
};




#define BASIC_INDEX_DRIVER_FIELDS   \
select_dir_ftype    * select_dir;   \
save_kwlist_ftype   * save_kwlist;  \
load_kwlist_ftype   * load_kwlist;  \
free_driver_ftype   * free_driver;  \
int                   type_id;


struct basic_index_driver_struct {
  BASIC_INDEX_DRIVER_FIELDS;
};




/*****************************************************************/



void  	 	  	   basic_driver_init(basic_driver_type * );
void 	 	  	   basic_driver_assert_cast(const basic_driver_type * );
basic_driver_type 	 * basic_driver_safe_cast(void * );

void 	 		   basic_driver_index_init(basic_driver_index_type * );
void 	 		   basic_driver_index_assert_cast(const basic_driver_index_type * );
basic_driver_index_type  * basic_driver_index_safe_cast(void * );

#ifdef __cplusplus
}
#endif
#endif
