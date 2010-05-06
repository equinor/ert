#ifndef __ENKF_NODE_H__
#define __ENKF_NODE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_config_node.h>
#include <arg_pack.h>
#include <ecl_kw.h>
#include <ecl_file.h>
#include <ecl_sum.h>
#include <fortio.h>
#include <enkf_serialize.h>
#include <buffer.h>
#include <msg.h>
#include <matrix.h>
#include <active_list.h>

/**********************************/


typedef void (matrix_serialize_ftype)   (const void * , const active_list_type *  ,       matrix_type * , int , int);
typedef void (matrix_deserialize_ftype) (      void * , const active_list_type *  , const matrix_type * , int , int);


/* Return value is the number of elements added = serial_size */
typedef void   	      (set_data_ftype)          (void *       	       ,  /* Node object     		*/
						 const void *);           /* The data - which is written in on the node. */

                                                 


typedef void   	      (ecl_write_ftype)         (const void *  ,   /* Node object */
						 const char *  ,   /* Directory to write to. */
                                                 const char *  ,   /* Filename - can be NULL. */
                                                 fortio_type *);   /* fortio inistance for writing elements in restart files. */

typedef void   	      (fload_ftype)                	(      void *  , const char *);
typedef void   	      (load_ftype)                	(      void *  , buffer_type * , int);
typedef bool   	      (store_ftype)                	(const void *  , buffer_type * , int , bool);


typedef void          (set_inflation_ftype)             (void *       ,   
                                                         const void * ,      /* Node object with the ensemble standard deviation. */
                                                         const void * );     /* Node object with the minimum standard deviation - supplied by the user. */



typedef double        (user_get_ftype)                  (void * , const char * , bool *);
typedef void * 	      (alloc_ftype)                	(const void *);
typedef bool   	      (initialize_ftype)     	   	(      void *  , int);
typedef bool   	      (ecl_load_ftype)            	(void *  , const char * , const ecl_sum_type * , const ecl_file_type * , int);
typedef void          (realloc_data_ftype)	   	(void * );
typedef void          (free_data_ftype)	           	(void * );
typedef void   	      (node_free_ftype)       	   	(      void *);
typedef void   	      (clear_ftype)      	   	(      void *);
typedef void   	      (node_copy_ftype)      	   	(const void * , void *);
typedef void   	      (isqrt_ftype)      	   	(      void *);
typedef void   	      (scale_ftype)      	   	(      void *  , double);
typedef void   	      (iadd_ftype)       	   	(      void *  , const void *);
typedef void   	      (imul_ftype)       	   	(      void *  , const void *);
typedef void   	      (iaddsqr_ftype)    	   	(      void *  , const void *);
typedef void          (ensemble_mulX_vector_ftype) 	(      void *  , int , const void ** , const double *);


typedef enum {alloc_func       	   	    = 0, 
	      ecl_write_func   	   	    = 1,
	      ecl_load_func                 = 2,
	      fread_func       	   	    = 3,
	      fwrite_func      	   	    = 4,
	      copy_func       	   	    = 5,
	      initialize_func  	   	    = 6,
	      free_func        	   	    = 7,
	      free_data_func   	   	    = 8 ,    
              clear_serial_state_func       =  9,  
	      matrix_serialize              = 10,
	      matrix_deserialize            = 11} node_function_type;
	      

typedef void          (enkf_node_ftype1)           (enkf_node_type *);
typedef void          (enkf_node_ftype_NEW)        (enkf_node_type * , arg_pack_type * );


double           enkf_node_user_get(enkf_node_type *  , const char * , bool * );
enkf_node_type * enkf_node_alloc(const enkf_config_node_type *);
enkf_node_type * enkf_node_alloc_with_data(const enkf_config_node_type * config , void * data);
enkf_node_type * enkf_node_copyc(const enkf_node_type * );
/*
  The enkf_node_free() function declaration is in the enkf_config_node.h header,
  because the enkf_config_node needs to know how to free the min_std node.

  void             enkf_node_free(enkf_node_type *enkf_node);
*/
void             enkf_node_free_data(enkf_node_type * );
void             enkf_node_free__(void *);
void             enkf_initialize    (enkf_node_type * , int);
bool             enkf_node_include_type(const enkf_node_type * , int );
void           * enkf_node_value_ptr(const enkf_node_type * );
enkf_impl_type   enkf_node_get_impl_type(const enkf_node_type * );
enkf_var_type    enkf_node_get_var_type(const enkf_node_type * );
void             enkf_node_clear_serial_state(enkf_node_type * );
void             enkf_node_matrix_serialize(enkf_node_type *enkf_node , const active_list_type * active_list , matrix_type * A , int row_offset , int column);
void             enkf_node_matrix_deserialize(enkf_node_type *enkf_node , const active_list_type * active_list , const matrix_type * A , int row_offset , int column);

bool             enkf_node_ecl_load  (enkf_node_type *, const char * , const ecl_sum_type * , const ecl_file_type * , int, int );
void             enkf_node_ecl_load_static  (enkf_node_type *, const ecl_kw_type * , int , int);
void             enkf_node_ecl_write (const enkf_node_type *, const char * , fortio_type * , int);
bool             enkf_node_initialize(enkf_node_type *enkf_node , int);
void             enkf_node_printf(const enkf_node_type *);
bool             enkf_node_fwrite (enkf_node_type * , FILE * stream, bool , int , int , state_enum);
void             enkf_node_clear     (enkf_node_type *);
void             enkf_node_fread  (enkf_node_type * , FILE * stream , int , int , state_enum);

void             enkf_node_fload( enkf_node_type * enkf_node , const char * filename );
void             enkf_node_load(enkf_node_type *enkf_node , buffer_type * buffer , int report_step , int iens , state_enum state);
bool             enkf_node_store(enkf_node_type *enkf_node , buffer_type * buffer , bool internal_state , int report_step , int iens , state_enum state);


/*****************************************************************/
/* Function callbacks */
ecl_write_ftype * enkf_node_get_func_pointer( const enkf_node_type * node );



void   enkf_node_set_inflation( enkf_node_type * inflation , const enkf_node_type * std , const enkf_node_type * min_std);
void   enkf_node_sqrt(enkf_node_type *enkf_node);
void   enkf_node_scale(enkf_node_type *   , double );
void   enkf_node_iadd(enkf_node_type *    , const enkf_node_type * );
void   enkf_node_iaddsqr(enkf_node_type * , const enkf_node_type * );
void   enkf_node_imul(enkf_node_type *    , const enkf_node_type * );
void   enkf_node_invalidate_cache( enkf_node_type * node );
const  enkf_config_node_type * enkf_node_get_config(const enkf_node_type * );
const char     *  enkf_config_node_get_infile(const enkf_config_node_type * );
const char     *  enkf_node_get_key(const enkf_node_type * );
const char     *  enkf_node_get_swapfile(const enkf_node_type *);
bool         	  enkf_node_has_func(const enkf_node_type * , node_function_type );
bool              enkf_node_internalize(const enkf_node_type * , int );

void              enkf_node_upgrade_file_103( const char * path , const char * file , enkf_impl_type impl_type , int perc_complete , msg_type * msg);

#ifdef __cplusplus
}
#endif
#endif
