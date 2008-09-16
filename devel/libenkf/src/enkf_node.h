#ifndef __ENKF_NODE_H__
#define __ENKF_NODE_H__
#include <stdlib.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_config_node.h>
#include <void_arg.h>
#include <ecl_kw.h>
#include <ecl_block.h>
#include <ecl_sum.h>
#include <fortio.h>

/**********************************/




#define NODE_STD_FIELDS \
int internal_offset;

/* Return value is the number of elements added = serial_size */
typedef int   	      (serialize_ftype)      	(const void * ,  /* Node object     	    */ 
                                                 int          ,  /* Internal offset 	    */   
                                                 size_t       ,  /* Size of serial buffer   */
                                                 double *     ,  /* Serial data             */
                                                 size_t       ,  /* Stride of serial data   */ 
                                                 size_t       ,  /* Offset of serial data   */
                                                 bool *       ,  /* Return flag : complete? */
                                                 serial_state_type *);


/* Return value is the updated internal offset  - not relevant if complete*/
typedef int   	      (deserialize_ftype)       (void *       	, /* Node object     		*/
                                                 int          	, /* Internal offset 		*/         
                                                 size_t       	, /* Serial size (of this node) */
                                                 const double * , /* Serial buffer              */
                                                 size_t         , /* Stride of serial data 	*/
                                                 size_t         , /* Offset of serial data 	*/   
                                                 serial_state_type *); 



typedef double        (iget_ftype)                      (const void * , int);    /* All obejects should support an iget */
typedef int           (get_index_ftype)                 (const void *); 
typedef void * 	      (alloc_ftype)                	(const void *);
typedef void   	      (fread_ftype)                	(      void *  , FILE *);
typedef bool   	      (fwrite_ftype)               	(const void *  , FILE *);
typedef void   	      (initialize_ftype)     	   	(      void *  , int);
typedef void   	      (ecl_write_ftype)            	(const void *  , const char *);
typedef void   	      (ecl_load_ftype)            	(void *  , const char * , const char *, const ecl_sum_type * , const ecl_block_type * , int);
typedef void          (realloc_data_ftype)	   	(void * );
typedef void          (free_data_ftype)	           	(void * );
typedef void   	      (free_ftype)       	   	(      void *);
typedef void   	      (clear_ftype)      	   	(      void *);
typedef void * 	      (copyc_ftype)      	   	(const void *);
typedef void   	      (isqrt_ftype)      	   	(      void *);
typedef void   	      (scale_ftype)      	   	(      void *  , double);
typedef void   	      (iadd_ftype)       	   	(      void *  , const void *);
typedef void   	      (imul_ftype)       	   	(      void *  , const void *);
typedef void   	      (iaddsqr_ftype)    	   	(      void *  , const void *);
typedef void          (ensemble_mulX_vector_ftype) 	(      void *  , int , const void ** , const double *);
typedef void          (ensemble_fprintf_results_ftype)  (const void ** , int , const char *);


typedef enum {alloc_func       	   	    = 0, 
	      ecl_write_func   	   	    = 1,
	      ecl_load_func                 = 2,
	      fread_func       	   	    = 3,
	      fwrite_func      	   	    = 4,
	      copyc_func       	   	    = 5,
	      initialize_func  	   	    = 6,
	      serialize_func   	   	    = 7,
	      deserialize_func 	   	    = 8,
	      free_func        	   	    = 9,
	      free_data_func   	   	    = 10,    
	      ensemble_fprintf_results_func = 11,
              clear_serial_state_func       = 12} node_function_type;
	      

	      

typedef struct enkf_node_struct enkf_node_type;

typedef void          (enkf_node_ftype1)           (enkf_node_type *);
typedef void          (enkf_node_ftype_NEW)        (enkf_node_type * , void_arg_type * );

enkf_node_type * enkf_node_alloc(const enkf_config_node_type *);
enkf_node_type * enkf_node_alloc_static(const char *);
enkf_node_type * enkf_node_copyc(const enkf_node_type * );
void             enkf_node_free(enkf_node_type *enkf_node);
void             enkf_node_free_data(enkf_node_type * );
void             enkf_node_free__(void *);
void             enkf_initialize    (enkf_node_type * , int);
bool             enkf_node_include_type(const enkf_node_type * , int );
void           * enkf_node_value_ptr(const enkf_node_type * );
enkf_impl_type   enkf_node_get_impl_type(const enkf_node_type * );
enkf_var_type    enkf_node_get_var_type(const enkf_node_type * );
void             enkf_node_clear_serial_state(enkf_node_type * );
void             enkf_node_deserialize(enkf_node_type * , double * , size_t );

void             enkf_node_ecl_load  (enkf_node_type *, const char * , const char * , const ecl_sum_type * , const ecl_block_type * , int);
void             enkf_node_ecl_load_static  (enkf_node_type *, const ecl_kw_type * , int);
void             enkf_node_ecl_write (const enkf_node_type *, const char *);
void             enkf_node_initialize(enkf_node_type *enkf_node , int);
void             enkf_node_printf(const enkf_node_type *);
bool              enkf_node_fwrite (enkf_node_type * , FILE * stream, int , state_enum);
int              enkf_node_serialize(enkf_node_type * , size_t , double * , size_t , size_t , bool *);
void             enkf_node_clear     (enkf_node_type *);
void             enkf_node_fread  (enkf_node_type * , FILE * stream , int , state_enum);
void             enkf_node_ensemble_fprintf_results(const enkf_node_type ** , int , int , const char * );

void   enkf_node_scale(enkf_node_type *   , double );
void   enkf_node_iadd(enkf_node_type *    , const enkf_node_type * );
void   enkf_node_iaddsqr(enkf_node_type * , const enkf_node_type * );
void   enkf_node_imul(enkf_node_type *    , const enkf_node_type * );
const  enkf_config_node_type * enkf_node_get_config(const enkf_node_type * );
const char     *  enkf_node_get_eclfile_ref(const enkf_node_type * );
const char     *  enkf_node_get_key_ref(const enkf_node_type * );
const char     *  enkf_node_get_swapfile(const enkf_node_type *);
bool         	  enkf_node_has_func(const enkf_node_type * , node_function_type );
void              enkf_node_load_static_ecl_kw(enkf_node_type *, const ecl_kw_type *);
void              enkf_node_ecl_write_fortio(const enkf_node_type * , fortio_type * , bool , enkf_impl_type);
#endif
