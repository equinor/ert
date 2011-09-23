#ifndef __ANALYSIS_TABLE_H__
#define __ANALYSIS_TABLE_H__

#ifdef __cplusplus
extern "C" {
#endif


#include <matrix.h>

#define ANALYSIS_NEED_ED              1
#define ANALYSIS_NEED_RANDROT         2
#define ANALYSIS_USE_A                4       







typedef void (analysis_initX_ftype)       (void * module_data , 
                                           matrix_type * X , 
                                           matrix_type * S , 
                                           matrix_type * R , 
                                           matrix_type * innov , 
                                           matrix_type * E , 
                                           matrix_type *D , 
                                           matrix_type * randrot); 

typedef bool (analysis_set_int_ftype)       (void * module_data , const char * flag , int value);
typedef bool (analysis_set_double_ftype)    (void * module_data , const char * var , double value);
typedef bool (analysis_set_string_ftype)    (void * module_data , const char * var , const char * value);
typedef void   (analysis_free_ftype) (void * );
typedef void * (analysis_alloc_ftype) ( );


  typedef void (analysis_init_update_ftype) (void * module_data, 
                                             const matrix_type * S , 
                                             const matrix_type * R , 
                                             const matrix_type * innov , 
                                             const matrix_type * E , 
                                             const matrix_type * D);
  
  typedef void (analysis_complete_update_ftype) (void * module_data );
  
  typedef bool (analysis_get_option_ftype) (void * module_data , long option);


/*****************************************************************/

#define EXTERNAL_MODULE_TABLE "analysis_table"


typedef struct {
  const char                     * name;
  analysis_initX_ftype           * initX;
  
  analysis_init_update_ftype     * init_update;
  analysis_complete_update_ftype * complete_update;

  analysis_free_ftype            * freef;
  analysis_alloc_ftype           * alloc;

  analysis_set_int_ftype         * set_int;
  analysis_set_double_ftype      * set_double;
  analysis_set_string_ftype      * set_string;
  analysis_get_option_ftype      * get_option;
} analysis_table_type;


#ifdef __cplusplus
}
#endif
#endif
