#ifndef __ANALYSIS_TABLE_H__
#define __ANALYSIS_TABLE_H__

#ifdef __cplusplus
extern "C" {
#endif


#include <matrix.h>




typedef void (analysis_initX_ftype)      (void * module_data , matrix_type * X , matrix_type * S , matrix_type * R , matrix_type * innov , matrix_type * E , matrix_type *D); 
typedef bool (analysis_set_flag_ftype)   (void * module_data , const char * flag , int value);
typedef bool (analysis_set_var_ftype)    (void * module_data , const char * var , double value);
typedef bool (analysis_set_string_ftype) (void * module_data , const char * var , const char * value);

  typedef void   (analysis_free_ftype) (void * );
  typedef void * (analysis_alloc_ftype) ( );

/*****************************************************************/

#define EXTERNAL_MODULE_TABLE "analysis_table"


typedef struct {
  const char                 * name;
  analysis_initX_ftype       * initX;

  analysis_free_ftype        * freef;
  analysis_alloc_ftype       * alloc;

  analysis_set_flag_ftype    * set_flag;
  analysis_set_var_ftype     * set_var;
  analysis_set_string_ftype  * set_string;
} analysis_table_type;


#ifdef __cplusplus
}
#endif
#endif
