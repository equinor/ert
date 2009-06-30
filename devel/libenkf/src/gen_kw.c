#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <util.h>
#include <gen_kw_config.h>
#include <gen_kw.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>
#include <enkf_macros.h>
#include <subst.h>
#include <buffer.h>
#include <matrix.h>


GET_DATA_SIZE_HEADER(gen_kw);


struct gen_kw_struct {
  int                       __type_id;
  const gen_kw_config_type *config;
  scalar_type              *scalar;
  subst_list_type          *subst_list;
};

/*****************************************************************/

void gen_kw_free_data(gen_kw_type *gen_kw) {
  scalar_free_data(gen_kw->scalar);
}



void gen_kw_free(gen_kw_type *gen_kw) {
  scalar_free(gen_kw->scalar);
  subst_list_free(gen_kw->subst_list);
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



gen_kw_type * gen_kw_alloc(const gen_kw_config_type * config) {
  gen_kw_type * gen_kw  = util_malloc(sizeof *gen_kw , __func__);
  gen_kw->config = config;
  gen_kw->scalar = scalar_alloc(gen_kw_config_get_scalar_config( config ));
  gen_kw->__type_id  = GEN_KW;
  gen_kw->subst_list = subst_list_alloc();  
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




bool gen_kw_store(const gen_kw_type *gen_kw , buffer_type * buffer,  bool internal_state) {
  buffer_fwrite_int( buffer , GEN_KW );
  scalar_buffer_fsave(gen_kw->scalar , buffer , internal_state);
  return true;
}



/**
   As of 17/03/09 (svn 1811) MULTFLT has been depreceated, and GEN_KW
   has been inserted as a 'drop-in-replacement'. This implies that
   existing storage labeled with implemantation type 'MULTFLT' should
   be silently 'upgraded' to 'GEN_KW'.
*/


void gen_kw_load(gen_kw_type * gen_kw , buffer_type * buffer) {
  enkf_impl_type file_type;
  file_type = buffer_fread_int(buffer);
  if ((file_type == GEN_KW) || (file_type == MULTFLT))
    scalar_buffer_fload( gen_kw->scalar , buffer);
}

void gen_kw_upgrade_103( const char * filename ) {
  FILE * stream            = util_fopen( filename , "r");
  enkf_impl_type impl_type = util_fread_int( stream );
  int size                 = util_fread_int( stream );
  double * data            = util_malloc( size * sizeof * data , __func__ ); 
  util_fread( data , sizeof * data , size , stream , __func__);
  fclose( stream );
  
  {
    buffer_type * buffer = buffer_alloc( 100 );
    buffer_fwrite_time_t( buffer , time(NULL));
    buffer_fwrite_int( buffer , impl_type );
    buffer_fwrite(buffer , data , sizeof * data    ,size);
    buffer_store( buffer , filename);
    buffer_free( buffer );
  }
  free( data );
}


void gen_kw_truncate(gen_kw_type * gen_kw) {
  scalar_truncate( gen_kw->scalar );  
}



bool gen_kw_initialize(gen_kw_type *gen_kw, int iens) {
  scalar_sample(gen_kw->scalar);  
  return true;
}



int gen_kw_serialize(const gen_kw_type *gen_kw , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(gen_kw->scalar , serial_state , serial_offset , serial_vector);
}


void gen_kw_deserialize(gen_kw_type *gen_kw , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  scalar_deserialize(gen_kw->scalar , serial_state , serial_vector);
}



void gen_kw_matrix_serialize(const gen_kw_type *gen_kw , const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  scalar_matrix_serialize(gen_kw->scalar , active_list , A , row_offset , column);
}


void gen_kw_matrix_deserialize(gen_kw_type *gen_kw , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  scalar_matrix_deserialize(gen_kw->scalar , active_list , A , row_offset , column);
}





/**
  This function takes an ensmeble of gen_kw instances, and allocates
  two new instances to hold the mean and standard deviation
  respectively. The return values are returned by reference.
*/
ALLOC_STATS_SCALAR(gen_kw)  



void gen_kw_filter_file(const gen_kw_type * gen_kw , const char * target_file) {
  const char * template_file = gen_kw_config_get_template_ref(gen_kw->config);
  if (template_file != NULL) {
    const int size               = gen_kw_config_get_data_size(gen_kw->config);
    const double * output_data   = scalar_get_output_ref(gen_kw->scalar);

    int ikw;
    
    gen_kw_output_transform(gen_kw);
    for (ikw = 0; ikw < size; ikw++) {
      const char * key = gen_kw_config_get_tagged_name(gen_kw->config , ikw);      
      subst_list_insert_owned_ref(gen_kw->subst_list , key , util_alloc_sprintf("%g" , output_data[ikw]));
    }
    
    subst_list_filter_file( gen_kw->subst_list , template_file , target_file);
  } else 
    util_abort("%s: internal error - tried to filter gen_kw instance without template file.\n",__func__);
}


void gen_kw_ecl_write(const gen_kw_type * gen_kw , const char * run_path , const char * base_file , fortio_type * fortio) {
  char * target_file = util_alloc_filename( run_path , base_file  , NULL);
  gen_kw_filter_file(gen_kw , target_file);
  free( target_file );
}



void gen_kw_export(const gen_kw_type * gen_kw , int * _size , char ***_kw_list , double **_output_values) {
  gen_kw_output_transform(gen_kw);

  *_kw_list       = gen_kw_config_get_name_list(gen_kw->config);
  *_size          = gen_kw_config_get_data_size(gen_kw->config);
  *_output_values = (double *) scalar_get_output_ref(gen_kw->scalar);

}


void gen_kw_ensemble_fprintf_results(const gen_kw_type ** ensemble, int ens_size , const char * filename) {
  char    ** kw_list = gen_kw_config_get_name_list(ensemble[0]->config);
  int        size    = gen_kw_config_get_data_size(ensemble[0]->config);
  int        * index = util_malloc(ens_size * sizeof * index , __func__);

  double  ** data    = util_malloc(size * sizeof * data, __func__);
  for(int i=0; i<size; i++)
    data[i] = util_malloc(ens_size * sizeof * data[i], __func__); 


  for (int iens = 0; iens < ens_size; iens++) {
    const double * scalar_data = scalar_get_output_ref(ensemble[iens]->scalar);
    for (int i = 0; i < size; i++) {
      data[i][iens] = scalar_data[i];
    }
    index[iens] = iens;
  }
  
  {
    FILE * stream = util_fopen(filename , "w");
    enkf_util_fprintf_data( index , (const double **) data, "Member #" , (const char **) kw_list, ens_size, size, true, stream);
    fclose(stream);
  }

  for(int i=0; i<size; i++)
    free(data[i]);
  free(data);
  free(index);
}


const char * gen_kw_get_name(const gen_kw_type * gen_kw, int kw_nr) {
  return  gen_kw_config_get_name(gen_kw->config , kw_nr);
}



/**
   Will return 0.0 on invalid input, and set valid -> false. It is the
   responsibility of the calling scope to check valid.
*/
double gen_kw_user_get(const gen_kw_type * gen_kw, const char * key , bool * valid) {
  const bool internal_value = false;
  int index = gen_kw_config_get_index(gen_kw->config , key);
  if (index >= 0) {
    *valid = true;
    return scalar_iget_double(gen_kw->scalar , internal_value , index);
  } else {
    *valid = false;
    fprintf(stderr,"** Warning:could not lookup key:%s in gen_kw instance \n",key);
    return 0.0;
  }
}




/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
SAFE_CAST(gen_kw , GEN_KW);
MATH_OPS_SCALAR(gen_kw);
VOID_ALLOC(gen_kw);
VOID_REALLOC_DATA(gen_kw);
VOID_SERIALIZE (gen_kw);
VOID_DESERIALIZE (gen_kw);
VOID_INITIALIZE(gen_kw);
VOID_FREE_DATA(gen_kw)
VOID_COPYC  (gen_kw)
VOID_FREE   (gen_kw)
VOID_ECL_WRITE(gen_kw)
VOID_USER_GET(gen_kw)
VOID_FPRINTF_RESULTS(gen_kw)
VOID_STORE(gen_kw)
VOID_LOAD(gen_kw)
VOID_MATRIX_SERIALIZE(gen_kw)
VOID_MATRIX_DESERIALIZE(gen_kw)
