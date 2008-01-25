#ifndef __PGBOX_CONFIG_H__
#define __PGBOX_CONFIG_H__
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <field_config.h>
#include <field.h>
#include <ecl_grid.h>
#include <void_arg.h>

typedef double (pgfilter_ftype) (const double * , const void_arg_type *);




typedef struct pgbox_config_struct pgbox_config_type;

struct pgbox_config_struct {
  CONFIG_STD_FIELDS;
  int               stride;

  const void_arg_type * arg;
  int                 * index_list;
  pgfilter_ftype      * filter;
  bool                  write_compressed;
  char                * target_key;
};



/*****************************************************************/
const char        * pgbox_config_get_target_key(const pgbox_config_type * );
bool 		    pgbox_config_write_compressed(const pgbox_config_type * );
int  		    pgbox_config_get_byte_size(const pgbox_config_type * );
pgbox_config_type * pgbox_config_alloc(const ecl_grid_type * grid , 
				       pgfilter_ftype * , 
				       const void_arg_type * arg, 
				       int stride , 
				       int i1 , int i2 , int j1 , int j2 , int k1 , int k2, field_config_type * target_config);


void                pgbox_config_apply(const pgbox_config_type * , const double * , field_type * );
void                pgbox_config_free(pgbox_config_type *  );

/*****************************************************************/
VOID_FREE_CONFIG_HEADER(pgbox);
GET_DATA_SIZE_HEADER(pgbox);

#endif
