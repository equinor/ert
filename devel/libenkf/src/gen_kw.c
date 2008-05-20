#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <enkf_state.h>
#include <util.h>
#include <gen_kw_config.h>
#include <gen_kw.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>


#define  DEBUG
#define  TARGET_TYPE GEN_KW
#include "enkf_debug.h"


GET_DATA_SIZE_HEADER(gen_kw);


struct gen_kw_struct {
  DEBUG_DECLARE
  const gen_kw_config_type *config;
  scalar_type              *scalar;
};

/*****************************************************************/

void gen_kw_free_data(gen_kw_type *gen_kw) {
  scalar_free_data(gen_kw->scalar);
}



void gen_kw_free(gen_kw_type *gen_kw) {
  scalar_free(gen_kw->scalar);
  free(gen_kw);
}



void gen_kw_realloc_data(gen_kw_type *gen_kw) {
  scalar_realloc_data(gen_kw->scalar);
}


void gen_kw_output_transform(const gen_kw_type * gen_kw) {
  scalar_transform(gen_kw->scalar);
}

void gen_kw_set_data(gen_kw_type * gen_kw , const double * data) {
  scalar_set_data(gen_kw->scalar , data);
}


void gen_kw_get_data(const gen_kw_type * gen_kw , double * data) {
  scalar_get_data(gen_kw->scalar , data);
}

void gen_kw_get_output_data(const gen_kw_type * gen_kw , double * output_data) {
  scalar_get_output_data(gen_kw->scalar , output_data);
}


const double * gen_kw_get_data_ref(const gen_kw_type * gen_kw) {
  return scalar_get_data_ref(gen_kw->scalar);
}


const double * gen_kw_get_output_ref(const gen_kw_type * gen_kw) {
  return scalar_get_output_ref(gen_kw->scalar);
}



gen_kw_type * gen_kw_alloc(const gen_kw_config_type * config) {
  gen_kw_type * gen_kw  = malloc(sizeof *gen_kw);
  gen_kw->config = config;
  gen_kw->scalar   = scalar_alloc(config->scalar_config); 
  DEBUG_ASSIGN(gen_kw)
  return gen_kw;
}


void gen_kw_clear(gen_kw_type * gen_kw) {
  scalar_clear(gen_kw->scalar);
}



gen_kw_type * gen_kw_copyc(const gen_kw_type *gen_kw) {
  gen_kw_type * new = gen_kw_alloc(gen_kw->config); 
  scalar_memcpy(new->scalar , gen_kw->scalar);
  return new; 
}


void gen_kw_fwrite(const gen_kw_type *gen_kw , FILE * stream) {
  DEBUG_ASSERT(gen_kw)
  enkf_util_fwrite_target_type(stream , GEN_KW);
  scalar_stream_fwrite(gen_kw->scalar , stream);
}


void gen_kw_fread(gen_kw_type * gen_kw , FILE * stream) {
  DEBUG_ASSERT(gen_kw)
  enkf_util_fread_assert_target_type(stream , GEN_KW , __func__);
  scalar_stream_fread(gen_kw->scalar , stream);
}



void gen_kw_swapout(gen_kw_type * gen_kw , FILE * stream) {
  gen_kw_fwrite(gen_kw , stream);
  gen_kw_free_data(gen_kw);
}


void gen_kw_swapin(gen_kw_type * gen_kw , FILE * stream) {
  gen_kw_realloc_data(gen_kw);
  gen_kw_fread(gen_kw , stream);
}


void gen_kw_truncate(gen_kw_type * gen_kw) {
  DEBUG_ASSERT(gen_kw)
  scalar_truncate( gen_kw->scalar );  
}



void  gen_kw_initialize(gen_kw_type *gen_kw, int iens) {
  DEBUG_ASSERT(gen_kw)
  scalar_sample(gen_kw->scalar);  
}



int gen_kw_serialize(const gen_kw_type *gen_kw , int internal_offset , size_t serial_data_size , double *serial_data , size_t ens_size , size_t offset, bool * complete) {
  DEBUG_ASSERT(gen_kw);
  return scalar_serialize(gen_kw->scalar , internal_offset , serial_data_size, serial_data , ens_size , offset , complete);
}


int gen_kw_deserialize(gen_kw_type *gen_kw , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  DEBUG_ASSERT(gen_kw);
  return scalar_deserialize(gen_kw->scalar , internal_offset , serial_size , serial_data , stride , offset);
}


/**
  This function takes an ensmeble of gen_kw instances, and allocates
  two new instances to hold the mean and standard deviation
  respectively. The return values are returned by reference.

  This function should probably be written as a macro...
*/

void gen_kw_alloc_stats(const gen_kw_type **gen_kw_ens , int ens_size , gen_kw_type ** _mean , gen_kw_type ** _std) {
  int iens;
  gen_kw_type * mean = gen_kw_copyc(gen_kw_ens[0]);
  gen_kw_type * std  = gen_kw_copyc(gen_kw_ens[0]);

  gen_kw_clear(mean);
  gen_kw_clear(std);

  for (iens = 0; iens < ens_size; iens++) {
    gen_kw_output_transform(gen_kw_ens[iens]);
    gen_kw_iadd(mean   , gen_kw_ens[iens]);
    gen_kw_iaddsqr(std , gen_kw_ens[iens]);
  }
  gen_kw_iscale(mean , 1.0 / ens_size);
  gen_kw_iscale(std  , 1.0 / ens_size);
  {
    gen_kw_type * tmp = gen_kw_copyc(mean);
    gen_kw_isqr(tmp);
    gen_kw_imul_add(std , -1.0 , tmp);
    gen_kw_free(tmp);
  }
  gen_kw_isqrt(std);

  *_mean = mean;
  *_std  = std;
}





void gen_kw_filter_file(const gen_kw_type * gen_kw , const char * target_file) {
  const char * template_file = gen_kw_config_get_template_ref(gen_kw->config);
  if (template_file != NULL) {
    const int size             = gen_kw_config_get_data_size(gen_kw->config);
    const double * output_data = scalar_get_output_ref(gen_kw->scalar);
    hash_type * kw_hash = hash_alloc();
    int ikw;
    
    gen_kw_output_transform(gen_kw);
    for (ikw = 0; ikw < size; ikw++)
      hash_insert_hash_owned_ref(kw_hash , gen_kw_config_get_name(gen_kw->config , ikw) , void_arg_alloc_double(output_data[ikw]) , void_arg_free__);
    
    util_filter_file(template_file , NULL , target_file , '<' , '>' , kw_hash , util_filter_warn0 );
    hash_free(kw_hash);
  } else 
    util_abort("%s: internal error - tried to filter gen_kw instance without template file.\n",__func__);
}


void gen_kw_ecl_write(const gen_kw_type * gen_kw , const char * target_file) {
  DEBUG_ASSERT(gen_kw)
  gen_kw_filter_file(gen_kw , target_file);

  /* 
     Esktra eksekverbar : sytem("xxxx ");
  */
}


void gen_kw_export(const gen_kw_type * gen_kw , int * _size , char ***_kw_list , double **_output_values) {
  gen_kw_output_transform(gen_kw);

  *_kw_list       = gen_kw_config_get_name_list(gen_kw->config);
  *_size          = gen_kw_config_get_data_size(gen_kw->config);
  *_output_values = (double *) scalar_get_output_ref(gen_kw->scalar);

}

#define PRINT_LINE(n,c,stream) { int _i; for (_i = 0; _i < (n); _i++) fputc(c , stream); fprintf(stream,"\n"); }
void gen_kw_ensemble_fprintf_results(const gen_kw_type ** ensemble, int ens_size , const char * filename) {
  const int float_width     =  9;
  const int float_precision =  5;
  char    ** kw_list = gen_kw_config_get_name_list(ensemble[0]->config);
  int        size    = gen_kw_config_get_data_size(ensemble[0]->config);
  int      * width   = util_malloc((size + 1) * sizeof * width , __func__);
  int        ikw , total_width;
  char     ** format;

  gen_kw_type * mean;
  gen_kw_type * std;

  gen_kw_alloc_stats(ensemble , ens_size , &mean , &std);
  format = util_malloc((size + 1) * sizeof * format , __func__);
  for (ikw = 0; ikw <= size; ikw++)
    format[ikw] = util_malloc(16 , __func__);

  width[0] = strlen("Member #|");
  total_width = width[0];
  for (ikw = 0; ikw < size; ikw++) {
    width[ikw + 1]  = util_int_max(strlen(kw_list[ikw]), 2 * float_width + 5) + 1;  /* Must accomodate A +/- B */
    width[ikw + 1] += ( 1 - (width[ikw + 1] & 1)); /* Ensure odd length */
    total_width += width[ikw + 1] + 1;
  }

  sprintf(format[0],"%%%dd" , width[0]);
  for (ikw = 0; ikw < size; ikw++)
    sprintf(format[ikw+1] , "%%%d.%df" , width[ikw+1] , float_precision);

  {
    FILE * stream = util_fopen(filename , "w");
    int iens;

    util_fprintf_string("Member #|" , width[0] , true , stream);
    for (ikw = 0; ikw < size; ikw++) {
      util_fprintf_string(kw_list[ikw] , width[ikw + 1] , true , stream);
      fprintf(stream , "|");
    }
    fprintf(stream , "\n");
    PRINT_LINE(total_width , '=' , stream);

    util_fprintf_string("Mean" , width[0] - 1 , true , stream);
    fprintf(stream , "|");
    {
      const double * mean_data = scalar_get_output_ref(mean->scalar);
      const double * std_data  = scalar_get_output_ref(std->scalar);
      for (ikw = 0; ikw < size; ikw++) {
	int w = (width[ikw + 1] - 5) / 2;
	util_fprintf_double(mean_data[ikw] , w , float_precision , stream);
	fprintf(stream , " +/- ");
	util_fprintf_double(std_data[ikw] , w , float_precision , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '-' , stream);
    for (iens = 0; iens < ens_size; iens++) {
      const double * data = scalar_get_output_ref(ensemble[iens]->scalar);
      util_fprintf_int(iens , width[0] - 1 , stream);
      fprintf(stream , "|");
      
      for (ikw = 0; ikw < size; ikw++) {
	util_fprintf_double(data[ikw] , width[ikw + 1] , float_precision , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '=' , stream);
    fclose(stream);
  }
  
  gen_kw_free(mean);
  gen_kw_free(std);
  free(width);
  
  for (ikw = 0; ikw <= size; ikw++)
    free(format[ikw]);
  free(format);
}
#undef PRINT_LINE

const char * gen_kw_get_name(const gen_kw_type * gen_kw, int kw_nr) {
  return  gen_kw_config_get_name(gen_kw->config , kw_nr);
}


/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

MATH_OPS_SCALAR(gen_kw);
VOID_ALLOC(gen_kw);
VOID_REALLOC_DATA(gen_kw);
VOID_SERIALIZE (gen_kw);
VOID_DESERIALIZE (gen_kw);
VOID_INITIALIZE(gen_kw);
VOID_FREE_DATA(gen_kw)
VOID_FWRITE (gen_kw)
VOID_FREAD  (gen_kw)
VOID_COPYC  (gen_kw)
VOID_FREE   (gen_kw)
VOID_ECL_WRITE(gen_kw)
VOID_FPRINTF_RESULTS(gen_kw)
