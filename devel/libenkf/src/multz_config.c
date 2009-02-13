#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>
#include <enkf_macros.h>
#include <multz_config.h>
#include <util.h>
#include <trans_func.h>


#define MULTZ_CONFIG_ID 78154



/*
  WARNING: Returns the multz_config object in a completely unitialized state.
*/
static multz_config_type * __multz_config_alloc_empty(int size ) {
  
  multz_config_type *multz_config = util_malloc(sizeof *multz_config , __func__);
  multz_config->__type_id = MULTZ_CONFIG_ID;
  multz_config->scalar_config = scalar_config_alloc_empty(size);
  
  multz_config->i1   	= util_malloc(size * sizeof *multz_config->i1      , __func__);
  multz_config->i2   	= util_malloc(size * sizeof *multz_config->i2      , __func__);
  multz_config->j1   	= util_malloc(size * sizeof *multz_config->j1      , __func__);
  multz_config->j2   	= util_malloc(size * sizeof *multz_config->j2      , __func__);
  multz_config->k    	= util_malloc(size * sizeof *multz_config->k       , __func__);
  multz_config->area 	= util_malloc(size * sizeof *multz_config->area    , __func__);

  return multz_config;
}






multz_config_type * multz_config_fscanf_alloc(const char * filename , int nx , int ny , int nz) {
  multz_config_type * config;
  FILE * stream = util_fopen(filename , "r");
  int size , line_nr;

  size = util_count_file_lines(stream);
  fseek(stream , 0L , SEEK_SET);
  config  = __multz_config_alloc_empty( size );
  line_nr = 0;
  do {
    int i1 = 1;
    int i2 = nx;
    int j1 = 1;
    int j2 = ny;
    int k;
    
    if (fscanf(stream , "%d" , &k) != 1) {
      fprintf(stderr,"%s: something wrong when reading: %s - aborting \n",__func__ , filename);
      abort();
    }
    config->k[line_nr]     = k;

    util_fscanf_int(stream, &i1);  
    util_fscanf_int(stream, &i2);
    util_fscanf_int(stream, &j1);
    util_fscanf_int(stream, &j2);

    config->i1[line_nr]    = util_int_max(i1 , 1);
    config->i2[line_nr]    = util_int_min(i2 , nx);
    config->j1[line_nr]    = util_int_max(j1 , 1);
    config->j2[line_nr]    = util_int_min(j2 , ny);
    config->area[line_nr]  = (config->i2[line_nr]- config->i1[line_nr] + 1) * (config->j2[line_nr]- config->j1[line_nr] + 1);
    
    scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
    line_nr++;
  } while ( line_nr < size );
  fclose(stream);
  return config;
}




void multz_config_ecl_write(const multz_config_type * config , const double *data , FILE *stream) {
  int ik;
  for (ik = 0; ik < multz_config_get_data_size(config); ik++) {

    fprintf(stream,"BOX\n   %5d %5d %5d %5d %5d %5d / \nMULTZ\n%d*%g /\nENDBOX\n\n\n" , 
	    config->i1[ik]   , config->i2[ik] , 
	    config->j1[ik]   , config->j2[ik] , 
	    config->k[ik]    , config->k[ik]  , 
	    config->area[ik] , data[ik]);
  }
  
}



void multz_config_free(multz_config_type * config) {
  free(config->j1);
  free(config->j2);
  free(config->i1);
  free(config->i2);
  free(config->k);
  free(config->area);
  scalar_config_free(config->scalar_config);
  free(config);
}



int multz_config_get_data_size(const multz_config_type * multz_config) {
  return scalar_config_get_data_size(multz_config->scalar_config);
}



char * multz_config_alloc_description(const multz_config_type * config, int multz_nr) {
  const int size = multz_config_get_data_size(config);
  if (multz_nr >= 0 && multz_nr < size) {
    char * description = util_malloc(48 * sizeof * description , __func__);
    sprintf(description , "k: %d  i: %d - %d  j: %d - %d" , config->k[multz_nr] , config->i1[multz_nr] , config->i2[multz_nr] , config->j1[multz_nr] , config->j2[multz_nr]);
    return description;
  } else {
    fprintf(stderr,"%s: asked for multz number:%d - valid interval: [0,%d] - aborting \n",__func__ , multz_nr , size - 1);
    abort();
  }
}


void multz_config_activate(multz_config_type * config , active_mode_type active_mode , void * active_config) {
  /*
   */
}


/*****************************************************************/
SAFE_CAST(multz_config , MULTZ_CONFIG_ID)
VOID_FREE(multz_config)
VOID_CONFIG_ACTIVATE(multz)
							 

