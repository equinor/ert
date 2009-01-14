#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <multz_config.h>
#include <multz.h>
#include <enkf_util.h>
#include <scalar.h>
#include <fortio.h>




/*****************************************************************/

GET_DATA_SIZE_HEADER(multz);

struct multz_struct {
  int                      __type_id;
  const multz_config_type *config;
  scalar_type             *scalar;
};

/*****************************************************************/
void multz_clear(multz_type * multz) {
  scalar_clear(multz->scalar);
}




void multz_output_transform(const multz_type * multz) {
  scalar_transform(multz->scalar);
}

void multz_set_data(multz_type * multz , const double * data) {
  scalar_set_data(multz->scalar , data);
}


void multz_get_data(const multz_type * multz , double * data) {
  scalar_get_data(multz->scalar , data);
}

void multz_get_output_data(const multz_type * multz , double * output_data) {
  scalar_get_output_data(multz->scalar , output_data);
}


const double * multz_get_data_ref(const multz_type * multz) {
  return scalar_get_data_ref(multz->scalar);
}


const double * multz_get_output_ref(const multz_type * multz) {
  return scalar_get_output_ref(multz->scalar);
}




void multz_realloc_data(multz_type *multz) {
  scalar_realloc_data(multz->scalar);
}


void multz_free_data(multz_type *multz) {
  scalar_free_data(multz->scalar);
}



multz_type * multz_alloc(const multz_config_type * multz_config) {
  multz_type * multz  = util_malloc(sizeof *multz , __func__);
  multz->config       = multz_config;
  multz->scalar       = scalar_alloc(multz_config->scalar_config); 
  multz->__type_id    = MULTZ;
  return multz;
}



multz_type * multz_copyc(const multz_type *multz) {
  multz_type * new = multz_alloc(multz->config); 
  scalar_memcpy(new->scalar , multz->scalar);
  return new; 
}


bool multz_fwrite(const multz_type *multz , FILE * stream) {
  enkf_util_fwrite_target_type(stream , MULTZ);
  scalar_stream_fwrite(multz->scalar , stream);
  return true;
}



void multz_fread(multz_type * multz , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , MULTZ);
  scalar_stream_fread(multz->scalar , stream);
}


void multz_ecl_write(const multz_type * multz , const char * eclfile, fortio_type * fortio) {
  {
    FILE * stream  = util_fopen(eclfile , "w");
    
    multz_output_transform(multz);
    multz_config_ecl_write(multz->config , multz_get_output_ref(multz) , stream);
    
    fclose(stream);
  }
}



void multz_free(multz_type *multz) {
  {
     scalar_free(multz->scalar);  
     free(multz);
  }
}


int multz_serialize(const multz_type *multz , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(multz->scalar , serial_state , serial_offset , serial_vector);
}


void multz_deserialize(multz_type *multz ,serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  scalar_deserialize(multz->scalar ,serial_state , serial_vector);
}


void multz_truncate(multz_type * multz) {
  scalar_truncate( multz->scalar );  
}



bool multz_initialize(multz_type *multz, int iens) {
  scalar_sample(multz->scalar);  
  return true;
}



multz_type * multz_alloc_mean(int ens_size , const multz_type **multz_ens) {
  int iens;
  multz_type * avg_multz = multz_copyc(multz_ens[0]);
  for (iens = 1; iens < ens_size; iens++) 
    multz_iadd(avg_multz , multz_ens[iens]);
  multz_iscale(avg_multz , 1.0 / ens_size);
  return avg_multz;
}



/*****************************************************************/


void multz_TEST() {
  const char * config_file = "/tmp/multz_config.txt";
  FILE * stream = util_fopen(config_file , "w");
  fprintf(stream , "1 1 10 1 10 0  CONST 1\n");
  fprintf(stream , "2 1 10         UNIFORM 0 1\n");
  fprintf(stream , "3     0        DUNIF   5 0 1\n");
  fclose(stream);
  
  {
    const int ens_size = 1000;
    char path[64];
    int iens;
    multz_config_type  * config    = multz_config_fscanf_alloc(config_file , 10, 10 ,10);
    multz_type        ** multz_ens = malloc(ens_size * sizeof * multz_ens);
    
    for (iens = 0; iens < ens_size; iens++) {
      multz_ens[iens] = multz_alloc(config);
      multz_initialize(multz_ens[iens] , 0);
      sprintf(path , "/tmp/%04d/MULTZ.INC" , iens + 1);
      util_make_path(path);
      multz_ecl_write(multz_ens[iens] , path , NULL);
      multz_truncate(multz_ens[iens]);
    }
  }
}


char * multz_alloc_description(const multz_type * multz , int multz_nr) {
  return multz_config_alloc_description(multz->config , multz_nr);
}





/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

SAFE_CAST(multz , MULTZ)
MATH_OPS_SCALAR(multz)
VOID_ALLOC(multz)
VOID_FREE(multz)
VOID_FREE_DATA(multz)
VOID_REALLOC_DATA(multz)
VOID_ECL_WRITE (multz)
VOID_FWRITE    (multz)
VOID_FREAD     (multz)
VOID_COPYC     (multz)
VOID_SERIALIZE(multz)
VOID_DESERIALIZE(multz)
VOID_TRUNCATE(multz)
VOID_SCALE(multz)
ENSEMBLE_MULX_VECTOR(multz)
ENSEMBLE_MULX_VECTOR_VOID(multz)
VOID_INITIALIZE(multz)



