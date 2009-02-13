#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <util.h>
#include <enkf_macros.h>
#include <multflt_config.h>
#include <multflt.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>
#include <fortio.h>



GET_DATA_SIZE_HEADER(multflt);


struct multflt_struct {
  int                        __type_id;
  const multflt_config_type *config;
  scalar_type               *scalar;
};

/*****************************************************************/

void multflt_free_data(multflt_type *multflt) {
  scalar_free_data(multflt->scalar);
}



void multflt_free(multflt_type *multflt) {
  scalar_free(multflt->scalar);
  free(multflt);
}



void multflt_realloc_data(multflt_type *multflt) {
  scalar_realloc_data(multflt->scalar);
}


void multflt_output_transform(const multflt_type * multflt) {
  scalar_transform(multflt->scalar);
}

void multflt_set_data(multflt_type * multflt , const double * data) {
  scalar_set_data(multflt->scalar , data);
}


void multflt_get_data(const multflt_type * multflt , double * data) {
  scalar_get_data(multflt->scalar , data);
}

void multflt_get_output_data(const multflt_type * multflt , double * output_data) {
  scalar_get_output_data(multflt->scalar , output_data);
}


const double * multflt_get_data_ref(const multflt_type * multflt) {
  return scalar_get_data_ref(multflt->scalar);
}


const double * multflt_get_output_ref(const multflt_type * multflt) {
  return scalar_get_output_ref(multflt->scalar);
}


multflt_type * multflt_alloc(const multflt_config_type * config) {
  multflt_type * multflt  = util_malloc(sizeof *multflt , __func__);
  multflt->config    	  = config;
  multflt->scalar    	  = scalar_alloc(multflt_config_get_scalar_config(config)); 
  multflt->__type_id 	  = MULTFLT;
  return multflt;
}


void multflt_clear(multflt_type * multflt) {
  scalar_clear(multflt->scalar);
}



multflt_type * multflt_copyc(const multflt_type *multflt) {
  multflt_type * new = multflt_alloc(multflt->config); 
  scalar_memcpy(new->scalar , multflt->scalar);
  return new; 
}

void  multflt_ecl_write(const multflt_type * multflt, const char * run_path , const char * eclfile , bool direct) {
  char * full_path = util_alloc_filename( run_path , eclfile  , NULL);
  FILE * stream    = util_fopen(full_path , "w");
  {
    const multflt_config_type *config = multflt->config;
    const int data_size       = multflt_config_get_data_size(config);
    const double *output_data = scalar_get_output_ref(multflt->scalar);
    int k;
    
    if (!direct) 
      multflt_output_transform(multflt);
    
    fprintf(stream , "MULTFLT\n");
    for (k=0; k < data_size; k++)
      fprintf(stream , " \'%s\'      %g  / \n", multflt_config_get_name( config , k) , output_data[k]);
    fprintf(stream , "/");
  }
  fclose(stream);
  free(full_path);
}



bool multflt_fwrite(const multflt_type *multflt , FILE * stream) {
  enkf_util_fwrite_target_type(stream , MULTFLT);
  scalar_stream_fwrite(multflt->scalar , stream);
  return true;
}


void multflt_fread(multflt_type * multflt , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , MULTFLT);
  scalar_stream_fread(multflt->scalar , stream);
}



void multflt_truncate(multflt_type * multflt) {
  scalar_truncate( multflt->scalar );  
}



bool  multflt_initialize(multflt_type *multflt, int iens) {
  scalar_sample(multflt->scalar);  
  return true;
}



int multflt_serialize(const multflt_type *multflt , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(multflt->scalar , serial_state , serial_offset , serial_vector);
}


void multflt_deserialize(multflt_type *multflt , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  scalar_deserialize(multflt->scalar , serial_state , serial_vector);
}




multflt_type * multflt_alloc_mean(int ens_size , const multflt_type **multflt_ens) {
  int iens;
  multflt_type * avg_multflt = multflt_copyc(multflt_ens[0]);
  for (iens = 1; iens < ens_size; iens++) 
    multflt_iadd(avg_multflt , multflt_ens[iens]);
  multflt_iscale(avg_multflt , 1.0 / ens_size);
  return avg_multflt;
}



#define PRINT_LINE(n,c,stream) { int _i; for (_i = 0; _i < (n); _i++) fputc(c , stream); fprintf(stream,"\n"); }
void multflt_ensemble_fprintf_results(const multflt_type ** ensemble, int ens_size , const char * filename) {
  const multflt_config_type * config = ensemble[0]->config;
  const int float_width     =  9;
  const int float_precision =  5;
  int        size    = multflt_config_get_data_size(ensemble[0]->config);
  int      * width   = util_malloc((size + 1) * sizeof * width , __func__);
  int        ikw , total_width;

  multflt_type * mean;
  multflt_type * std;

  multflt_alloc_stats(ensemble , ens_size , &mean , &std);
  width[0] = strlen("Member #|");
  total_width = width[0];
  for (ikw = 0; ikw < size; ikw++) {
    width[ikw + 1]  = util_int_max(strlen(multflt_config_get_name(config , ikw)), 2 * float_width + 5) + 1;  /* Must accomodate A +/- B */
    width[ikw + 1] += ( 1 - (width[ikw + 1] & 1)); /* Ensure odd length */
    total_width += width[ikw + 1] + 1;
  }

  {
    FILE * stream = util_fopen(filename , "w");
    int iens;

    util_fprintf_string("Member #|" , width[0] , true , stream);
    for (ikw = 0; ikw < size; ikw++) {
      util_fprintf_string(multflt_config_get_name(config , ikw) , width[ikw + 1] , center , stream);
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
	util_fprintf_double(mean_data[ikw] , w , float_precision , 'g' , stream);
	fprintf(stream , " +/- ");
	util_fprintf_double(std_data[ikw] , w , float_precision , 'g' , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '-' , stream);
    for (iens = 0; iens < ens_size; iens++) {
      const double * data = scalar_get_output_ref(ensemble[iens]->scalar);
      util_fprintf_int(iens + 1, width[0] - 1 , stream);
      fprintf(stream , "|");
      
      for (ikw = 0; ikw < size; ikw++) {
	util_fprintf_double(data[ikw] , width[ikw + 1] , float_precision , 'g' , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '=' , stream);
    fclose(stream);
  }
  
  multflt_free(mean);
  multflt_free(std);
  free(width);
}
#undef PRINT_LINE





/*****************************************************************/



const char * multflt_get_name(const multflt_type * multflt, int fault_nr) {
  return  multflt_config_get_name(multflt->config , fault_nr);
}



/**
   Will return 0.0 on invalid input, and set valid -> false. It is the
   responsibility of the calling scope to check valid.
*/
double multflt_user_get(const multflt_type * multflt, const char * key , bool * valid) {
  const bool internal_value = false;
  int index = multflt_config_get_index(multflt->config , key);
  if (index >= 0) {
    *valid = true;
    return scalar_iget_double(multflt->scalar , internal_value , index);
  } else {
    *valid = false;
    fprintf(stderr,"** Warning:could not lookup fault:%s in multflt instance.\n",key);
    return 0.0;
  }
}




MATH_OPS_SCALAR(multflt);
VOID_ALLOC(multflt);
VOID_SERIALIZE (multflt);
VOID_DESERIALIZE (multflt);
VOID_INITIALIZE(multflt);
/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
SAFE_CAST(multflt , MULTFLT)
VOID_FREE_DATA(multflt);
VOID_ECL_WRITE (multflt)
VOID_FWRITE (multflt)
VOID_FREAD  (multflt)
VOID_COPYC  (multflt)
VOID_FREE(multflt)
VOID_REALLOC_DATA(multflt)
ALLOC_STATS_SCALAR(multflt)
VOID_FPRINTF_RESULTS(multflt)
VOID_USER_GET(multflt)
