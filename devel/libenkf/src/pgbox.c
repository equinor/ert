#include <math.h>
#include <string.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <util.h>
#include <ecl_box.h>
#include <pgbox_config.h>
#include <pgbox.h>
#include <field.h>



#define  DEBUG
#define  TARGET_TYPE FIELD
#include "enkf_debug.h"


struct pgbox_struct {
  DEBUG_DECLARE
  const pgbox_config_type * config;
  field_type        	  * target_field;
  double            	  * data;
};


/*****************************************************************/


static void pgbox_realloc_data(pgbox_type * pgbox) {
  pgbox->data = util_malloc(pgbox_config_get_byte_size(pgbox->config) , __func__);
}





pgbox_type * pgbox_alloc(const pgbox_config_type * config) {
  pgbox_type * pgbox  = util_malloc(sizeof * pgbox , __func__);
  pgbox->config       = config;
  pgbox->target_field = NULL;
  pgbox->data = NULL;
  pgbox_realloc_data(pgbox);
  DEBUG_ASSIGN(pgbox)
  return pgbox;
}


void pgbox_set_target_field(pgbox_type * pgbox , field_type * target_field) {
  pgbox->target_field = target_field;
}


void pgbox_apply(pgbox_type * pgbox) {
  if (pgbox->target_field == NULL) {
    fprintf(stderr,"%s: must call pgbox_set_target_field() first - aborting \n",__func__);
    abort();
  }
  pgbox_config_apply(pgbox->config , pgbox->data , pgbox->target_field);
}



void pgbox_free_data(pgbox_type * pgbox) {
  DEBUG_ASSERT(pgbox);
  free(pgbox->data);
  pgbox->data = NULL;
}



void pgbox_free(pgbox_type * pgbox) {
  free(pgbox->data);
  free(pgbox);
}


void pgbox_fwrite(const pgbox_type * pgbox , FILE * stream) {
  const int data_size    = pgbox_config_get_data_size(pgbox->config);
  const int sizeof_ctype = sizeof * pgbox->data;
  bool  write_compressed = pgbox_config_write_compressed(pgbox->config);
  
  enkf_util_fwrite_target_type(stream , PGBOX);
  fwrite(&data_size               ,   sizeof  data_size        , 1 , stream);
  fwrite(&sizeof_ctype            ,   sizeof  sizeof_ctype     , 1 , stream);
  fwrite(&write_compressed        ,   sizeof  write_compressed , 1 , stream);
  if (write_compressed)
    util_fwrite_compressed(pgbox->data , sizeof_ctype * data_size , stream);
  else
    enkf_util_fwrite(pgbox->data    ,   sizeof_ctype , data_size , stream , __func__);
}



void pgbox_fread(pgbox_type * pgbox , FILE * stream) {
  int  data_size , sizeof_ctype;
  bool read_compressed;
  enkf_util_fread_assert_target_type(stream , PGBOX , __func__);
  fread(&data_size     	  , sizeof  data_size        , 1 , stream);
  fread(&sizeof_ctype 	  , sizeof  sizeof_ctype     , 1 , stream);
  fread(&read_compressed  , sizeof  read_compressed  , 1 , stream);
  if (read_compressed)
    util_fread_compressed((char *) &pgbox->data[0] , stream);
  else
    enkf_util_fread(pgbox->data , sizeof_ctype , data_size , stream , __func__);
}


pgbox_type * pgbox_copyc(const pgbox_type * src) {
  pgbox_type * target = pgbox_alloc(src->config);
  if (src->target_field != NULL) 
    target->target_field = field_copyc(src->target_field);
  if (src->data != NULL) {
    const int byte_size    = pgbox_config_get_byte_size(src->config);
    memcpy(target->data , src->data , byte_size);
  } else
    free(target->data);

  return target;
}




void pgbox_clear(pgbox_type * pgbox) {
  const int data_size          = pgbox_config_get_data_size(pgbox->config);   
  int i;
  for (i = 0; i < data_size; i++) 
    pgbox->data[i] = 0.0;
}




void pgbox_swapout(pgbox_type * pgbox , FILE * stream) {
  pgbox_fwrite(pgbox , stream);
  pgbox_free_data(pgbox);
}



void pgbox_swapin(pgbox_type * pgbox , FILE * stream) {
  pgbox_realloc_data(pgbox);
  pgbox_fread(pgbox  , stream);
}



int pgbox_deserialize(const pgbox_type * pgbox , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  const pgbox_config_type *config      = pgbox->config;
  const int                data_size   = pgbox_config_get_data_size(config);
  double *data;
  int new_internal_offset;

  data = &pgbox->data[internal_offset];
  new_internal_offset = enkf_util_deserialize(data , NULL , internal_offset , data_size , serial_size , serial_data , offset , stride);
  return new_internal_offset;
}




int pgbox_serialize(const pgbox_type *pgbox , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  const pgbox_config_type *config     = pgbox->config;
  const int                data_size  = pgbox_config_get_data_size(config);

  int elements_added;
  elements_added = enkf_util_serialize(pgbox->data , NULL , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
  return elements_added;
}



void pgbox_sample(pgbox_type *pgbox) {
  printf("%s: Warning not implemented ... \n",__func__);
}




/*****************************************************************/

VOID_FWRITE (pgbox)
VOID_FREAD  (pgbox)


MATH_OPS(pgbox)
VOID_ALLOC(pgbox)
VOID_FREE(pgbox)
VOID_FREE_DATA(pgbox)
ENSEMBLE_MULX_VECTOR(pgbox);
VOID_REALLOC_DATA(pgbox)
VOID_COPYC     (pgbox)
VOID_SWAPIN(pgbox)
VOID_SWAPOUT(pgbox)
VOID_SERIALIZE (pgbox);
VOID_DESERIALIZE (pgbox);

/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

VOID_FUNC      (pgbox_clear        , pgbox_type)
VOID_FUNC      (pgbox_sample       , pgbox_type)
