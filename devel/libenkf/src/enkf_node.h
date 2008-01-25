#ifndef __ENKF_NODE_H__
#define __ENKF_NODE_H__
#include <stdlib.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_config_node.h>

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
                                                 bool * );       /* Return flag : complete? */


/* Return value is the updated internal offset  - not relevant if complete*/
typedef int   	      (deserialize_ftype)       (void *       	, /* Node object     		*/
                                                 int          	, /* Internal offset 		*/         
                                                 size_t       	, /* Serial size (of this node) */
                                                 const double * , /* Serial buffer              */
                                                 size_t         , /* Stride of serial data 	*/
                                                 size_t );        /* Offset of serial data 	*/   


typedef void * 	      (fread_alloc_ftype)          (const void * , FILE *);
typedef void * 	      (alloc_ftype)                (const void *);
typedef void   	      (ecl_write_ftype)            (const void * , const char *);
typedef void   	      (fread_ftype)                (      void * , FILE *);
typedef void   	      (fwrite_ftype)               (const void * , FILE *);
typedef void          (swapin_ftype)      	   (void * , FILE *);
typedef void          (swapout_ftype)     	   (void * , FILE *);
typedef void   	      (sample_ftype)     	   (      void *);
typedef void   	      (free_ftype)       	   (      void *);
typedef void   	      (clear_ftype)      	   (      void *);
typedef void * 	      (copyc_ftype)      	   (const void *);
typedef void   	      (isqrt_ftype)      	   (      void *);
typedef void   	      (scale_ftype)      	   (      void * , double);
typedef void   	      (iadd_ftype)       	   (      void * , const void *);
typedef void   	      (imul_ftype)       	   (      void * , const void *);
typedef void   	      (iaddsqr_ftype)    	   (      void * , const void *);
typedef void          (ensemble_mulX_vector_ftype) (      void * , int , const void ** , const double *);

typedef struct enkf_node_struct enkf_node_type;

typedef void          (enkf_node_ftype1)        (enkf_node_type *);

enkf_node_type * enkf_node_alloc(const char * , const enkf_config_node_type *);


enkf_node_type * enkf_node_copyc(const enkf_node_type * );
enkf_node_type * enkf_node_alloc_old(const char *  	       ,  /*  1  */
				 const enkf_config_node_type *  ,  /*  2  */
				 alloc_ftype * 	       ,  /*  3  */    
				 ecl_write_ftype *     ,  /*  4  */
				 fread_ftype *         ,  /*  5  */
				 fwrite_ftype *        ,  /*  6  */
				 swapout_ftype *       ,  /*  7  */ 
				 swapin_ftype  *       ,  /*  8  */
				 copyc_ftype * 	       ,  /*  9  */
				 sample_ftype *	       ,  /* 10  */    
				 serialize_ftype    *  ,  /* 11  */
				 deserialize_ftype *   ,  /* 12  */
				 free_ftype);             /* 13  */ 

void             enkf_node_free(enkf_node_type *enkf_node);
void             enkf_node_free__(void *);
void             enkf_sample    (enkf_node_type *);
bool             enkf_node_include_type(const enkf_node_type * , int );
void           * enkf_node_value_ptr(const enkf_node_type * );
enkf_impl_type   enkf_node_get_impl_type(const enkf_node_type * );
enkf_var_type    enkf_node_get_var_type(const enkf_node_type * );
void             enkf_node_clear_serial_state(enkf_node_type * );
void             enkf_node_deserialize(enkf_node_type * , double * , size_t );

void             enkf_node_ecl_write (const enkf_node_type *, const char *);
/*void             enkf_node_ecl_read  (enkf_node_type * , const char *);*/
void             enkf_node_sample(enkf_node_type *enkf_node);
void             enkf_node_printf(const enkf_node_type *);
void             enkf_node_fwrite (const enkf_node_type * , FILE * stream);
int              enkf_node_serialize(enkf_node_type * , size_t , double * , size_t , size_t , bool *);
void             enkf_node_clear     (enkf_node_type *);
void             enkf_node_fread  (enkf_node_type * , FILE * stream);
const char     * enkf_node_get_key_ref(const enkf_node_type * );
bool enkf_node_swapped(const enkf_node_type *);
const char * enkf_node_get_swapfile(const enkf_node_type *);
void enkf_node_swapin(enkf_node_type * , const char * );
void enkf_node_swapout(enkf_node_type * , const char *);

void   enkf_node_scale(enkf_node_type *   , double );
void   enkf_node_iadd(enkf_node_type *    , const enkf_node_type * );
void   enkf_node_iaddsqr(enkf_node_type * , const enkf_node_type * );
void   enkf_node_imul(enkf_node_type *    , const enkf_node_type * );
const  enkf_config_node_type * enkf_node_get_config(const enkf_node_type * );

#endif
