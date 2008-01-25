#ifndef __ENKF_UTIL_H__
#define __ENKF_UTIL_H__
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <enkf_types.h>


/*****************************************************************/

#define VOID_FUNC_CONST(func,type) \
void func ## __(const void *void_arg) { \
   const type *arg = (const type *) void_arg; \
   func(arg);                             \
}

#define VOID_FUNC(func,type) \
void func ## __(void *void_arg) { \
   type *arg = (type *) void_arg; \
   func(arg);                     \
}



#define VOID_FUNC_HEADER(func) void func ## __(void *void_arg)
#define VOID_FUNC_HEADER_CONST(func) void func ## __(const void *void_arg)
#define VOID_SCALE_FUNC_HEADER(func) void func ## __(void *, double )
#define VOID_ADD_FUNC_HEADER(func) void func ## __(void *, const void *) 

/*****************************************************************/

#define SQRT_FUNC(prefix)                           \
void prefix ## _isqrt(void * void_arg) {            \
prefix ## _type *arg = (prefix ## _type *) void_arg;\
const prefix ## _config_type *config = arg->config; \
const int data_size = config->data_size;            \
int i;                                              \
for (i=0; i < data_size; i++)                       \
 arg->data[i] = sqrt(arg->data[i]);                 \
}
#define SQRT_FUNC_SCALAR(prefix)   void prefix ## _isqrt(void * void_arg) { scalar_isqrt(((prefix ## _type *) void_arg)->scalar); }
#define SQRT_FUNC_HEADER(prefix) void prefix ## _isqrt(void * )

/*****************************************************************/

#define SCALE_FUNC(prefix)                                                 \
void prefix ## _iscale(void *void_arg , double scale_factor) {             \
prefix ## _type *arg = (prefix ## _type *) void_arg;                       \
const prefix ## _config_type *config = arg->config; 			   \
const int data_size = config->data_size;                      			   \
int i;                                              			   \
for (i=0; i < data_size; i++)                            			   \
 arg->data[i] *= scale_factor;                  			   \
}
#define SCALE_FUNC_SCALAR(prefix)   void prefix ## _iscale(void * void_arg, double scale_factor) { scalar_iscale(((prefix ## _type *) void_arg)->scalar , scale_factor); }
#define SCALE_FUNC_HEADER(prefix) void prefix ## _iscale(void * , double)

/*****************************************************************/

#define RESET_FUNC(prefix)                                                 \
void prefix ## _ireset(void *void_arg) {             \
prefix ## _type *arg = (prefix ## _type *) void_arg;                       \
const prefix ## _config_type *config = arg->config; 			   \
const int data_size = config->data_size;                      			   \
int i;                                              			   \
for (i=0; i < data_size; i++)                            			   \
 arg->data[i] = 0;                               			   \
}
#define RESET_FUNC_SCALAR(prefix)   void prefix ## _ireset(void * void_arg) { scalar_ireset(((prefix ## _type *) void_arg)->scalar); }
#define RESET_FUNC_HEADER(prefix) void prefix ## _ireset(void *)

/*****************************************************************/

#define ADD_FUNC(prefix)                                                       \
void prefix ## _iadd(void *void_arg , const void *void_delta) {                \
      prefix ## _type *arg   = (prefix ## _type *)       void_arg;  	       \
const prefix ## _type *delta = (const prefix ## _type *) void_delta;	       \
const prefix ## _config_type *config = arg->config; 			       \
const int data_size = config->data_size;                      			       \
int i;                                              			       \
if (config != delta->config) {                                                 \
    fprintf(stderr,"%s:two multz object have different config objects - aborting \n",__func__);\
    abort();                                                                   \
}                                                                              \
for (i=0; i < data_size; i++)                            			       \
 arg->data[i] += delta->data[i];                                               \
}

#define ADD_FUNC_SCALAR(prefix)   void prefix ## _iadd(void *void_arg,  const void * void_factor) { scalar_iadd( ((prefix ## _type *) void_arg)->scalar , (( const prefix ## _type *) void_factor)->scalar); }
#define ADD_FUNC_HEADER(prefix) void prefix ## _iadd(void * , const void *)
/*****************************************************************/

#define MUL_ADD_FUNC(prefix)                                                                         \
void prefix ## _imul_add(void *void_arg , double scale_factor , const void *void_delta) {                \
      prefix ## _type *arg   = (prefix ## _type *)       void_arg;  	       \
const prefix ## _type *delta = (const prefix ## _type *) void_delta;	       \
const prefix ## _config_type *config = arg->config; 			       \
const int data_size = config->data_size;                      			       \
int i;                                              			       \
if (config != delta->config) {                                                 \
    fprintf(stderr,"%s:two multz object have different config objects - aborting \n",__func__);\
    abort();                                                                   \
}                                                                              \
for (i=0; i < data_size; i++)                            			       \
 arg->data[i] += scale_factor * delta->data[i];                                        \
}

#define MUL_ADD_FUNC_SCALAR(prefix)   void prefix ## _imul_add(void *void_arg,  double scale_factor , const void * void_factor) { scalar_imul_add( ((prefix ## _type *) void_arg)->scalar , scale_factor , (( const prefix ## _type *) void_factor)->scalar); }
#define MUL_ADD_FUNC_HEADER(prefix) void prefix ## _imul_add(void * , double , const void *)

/*****************************************************************/

#define SUB_FUNC(prefix)                                                       \
void prefix ## _isub(void *void_arg , const void *void_diff) {                \
      prefix ## _type *arg   = (prefix ## _type *)       void_arg;  	       \
const prefix ## _type *diff = (const prefix ## _type *) void_diff;	       \
const prefix ## _config_type *config = arg->config; 			       \
const int data_size = config->data_size;                      			       \
int i;                                              			       \
if (config != diff->config) {                                                 \
    fprintf(stderr,"%s:two multz object have different config objects - aborting \n",__func__);\
    abort();                                                                   \
}                                                                              \
for (i=0; i < data_size; i++)                            			       \
 arg->data[i] -= diff->data[i];                                               \
}

#define SUB_FUNC_SCALAR(prefix)   void prefix ## _isub(void *void_arg,  const void * void_factor) { scalar_isub( ((prefix ## _type *) void_arg)->scalar , (( const prefix ## _type *) void_factor)->scalar); }
#define SUB_FUNC_HEADER(prefix) void prefix ## _isub(void * , const void *)
/*****************************************************************/

#define MUL_FUNC(prefix)                                                       \
void prefix ## _imul(void *void_arg , const void *void_factor) {                \
      prefix ## _type *arg    = (prefix ## _type *)       void_arg;  	       \
const prefix ## _type *factor = (const prefix ## _type *) void_factor;	       \
const prefix ## _config_type *config = arg->config; 			       \
const int data_size = config->data_size;                      			       \
int i;                                              			       \
if (config != factor->config) {                                                 \
    fprintf(stderr,"%s:two multz object have different config objects - aborting \n",__func__);\
    abort();                                                                   \
}                                                                              \
for (i=0; i < data_size; i++)                            			       \
 arg->data[i] *= factor->data[i];                                               \
}
#define MUL_FUNC_SCALAR(prefix)   void prefix ## _imul(void *void_arg,  const void * void_factor) { scalar_imul( ((prefix ## _type *) void_arg)->scalar , (( const prefix ## _type *) void_factor)->scalar); }
#define MUL_FUNC_HEADER(prefix) void prefix ## _imul(void * , const void *)

/*****************************************************************/

#define ADDSQR_FUNC(prefix)                                                       \
void prefix ## _iaddsqr(void *void_arg , const void *void_delta) {                \
      prefix ## _type *arg   = (prefix ## _type *)       void_arg;  	       \
const prefix ## _type *delta = (const prefix ## _type *) void_delta;	       \
const prefix ## _config_type *config = arg->config; 			       \
const int data_size = config->data_size;                      			       \
int i;                                              			       \
if (config != delta->config) {                                                 \
    fprintf(stderr,"%s:two multz object have different config objects - aborting \n",__func__);\
    abort();                                                                   \
}                                                                              \
for (i=0; i < data_size; i++)                            			       \
 arg->data[i] += delta->data[i] * delta->data[i];                              \
}
#define ADDSQR_FUNC_SCALAR(prefix)   void prefix ## _iaddsqr(void *void_arg,  const void * void_factor) { scalar_iaddsqr( ((prefix ## _type *) void_arg)->scalar , (( const prefix ## _type *) void_factor)->scalar); }
#define ADDSQR_FUNC_HEADER(prefix) void prefix ## _iaddsqr(void * , const void *)


#define MATH_OPS(prefix) \
SQRT_FUNC   (prefix) \
SCALE_FUNC  (prefix) \
ADD_FUNC    (prefix) \
ADDSQR_FUNC (prefix) \
SUB_FUNC    (prefix) \
RESET_FUNC  (prefix) \
MUL_FUNC    (prefix) \
MUL_ADD_FUNC(prefix) 

#define MATH_OPS_SCALAR(prefix) \
SQRT_FUNC_SCALAR   (prefix) \
SCALE_FUNC_SCALAR  (prefix) \
ADD_FUNC_SCALAR    (prefix) \
ADDSQR_FUNC_SCALAR (prefix) \
SUB_FUNC_SCALAR    (prefix) \
RESET_FUNC_SCALAR  (prefix) \
MUL_FUNC_SCALAR    (prefix) \
MUL_ADD_FUNC_SCALAR(prefix) 



#define MATH_OPS_HEADER(prefix) \
SQRT_FUNC_HEADER   (prefix);  \
SCALE_FUNC_HEADER  (prefix);  \
ADD_FUNC_HEADER    (prefix);  \
ADDSQR_FUNC_HEADER (prefix);  \
SUB_FUNC_HEADER    (prefix);  \
RESET_FUNC_HEADER  (prefix);  \
MUL_FUNC_HEADER    (prefix);  \
MUL_ADD_FUNC_HEADER(prefix)

/*****************************************************************/

#define ENSEMBLE_MULX_VECTOR(prefix) \
void prefix ## _ensemble_mulX_vector(prefix ## _type *new , int ens_size , const prefix ## _type ** prefix ## _ensemble , const double *X_vector) { \
  int iens;              \
  prefix ## _clear(new); \
  for (iens=0; iens < ens_size; iens++)  \
     prefix ## _imul_add(new , X_vector[iens] , prefix ## _ensemble[iens]);\
}
#define ENSEMBLE_MULX_VECTOR_HEADER(prefix) void prefix ## _ensemble_mulX_vector(prefix ## _type *, int , const prefix ## _type ** , const double *);


#define ENSEMBLE_MULX_VECTOR_VOID(prefix) \
void prefix ## _ensemble_mulX_vector__(void *new , int ens_size , const void ** prefix ## _ensemble , const double *X_vector) { \
   prefix ## _ensemble_mulX_vector((prefix ## _type *) new , ens_size , (const prefix ## _type **) prefix ## _ensemble , X_vector); \
} 

#define ENSEMBLE_MULX_VECTOR_VOID_HEADER(prefix) void prefix ## _ensemble_mulX_vector__(void * , int  , const void ** , const  double *);

/*****************************************************************/


size_t enkf_util_serialize(const double * , const bool * , size_t , size_t , double * , size_t , size_t , int ,  bool * );
size_t enkf_util_deserialize(double *     , const bool * , size_t , size_t , size_t   , const double * , size_t , int );
void * enkf_util_calloc(int , int , const char * );
void * enkf_util_malloc (int , const char * );
void * enkf_util_realloc(void * , int , const char * );
FILE * enkf_util_fopen_r(const char * , const char * );
FILE * enkf_util_fopen_w(const char * , const char * );
void   enkf_util_fwrite(const void *, int , int , FILE *, const char * );
void   enkf_util_fread ( void *, int , int , FILE *, const char * );

double enkf_util_rand_normal(double , double );
void   enkf_util_fread_assert_target_type(FILE * , enkf_impl_type , const char *);
void   enkf_util_fwrite_target_type(FILE * , enkf_impl_type);

#endif
