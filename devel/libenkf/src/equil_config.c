#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <equil_config.h>
#include <enkf_util.h>
#include <scalar_config.h>



/*****************************************************************/

equil_config_type * equil_config_alloc(int size) {
  equil_config_type *config        = malloc(sizeof *config);

  config->data_size    	  	   = size;
                            
  config->datum_depth  	     	   = enkf_util_malloc(size * sizeof * config->datum_depth  	       , __func__);
  config->datum_P      	     	   = enkf_util_malloc(size * sizeof * config->datum_P      	       , __func__);
  config->oil_water_Pc 	     	   = enkf_util_malloc(size * sizeof * config->oil_water_Pc 	       , __func__);
  config->gas_oil_Pc   	     	   = enkf_util_malloc(size * sizeof * config->gas_oil_Pc   	       , __func__);
  config->live_oil_init_mode 	   = enkf_util_malloc(size * sizeof * config->live_oil_init_mode       , __func__);
  config->black_oil_wgas_init_mode = enkf_util_malloc(size * sizeof * config->black_oil_wgas_init_mode , __func__);
  config->init_accuracy            = enkf_util_malloc(size * sizeof * config->init_accuracy            , __func__);
  
  config->scalar_config            = scalar_config_alloc_empty(size * 2);
  /*
    The underlying scalar structure assumes that first comes woc, and
    then comes goc.
  */
  return config;
}


equil_config_type * equil_config_fscanf_alloc(const char * config_file) {
  FILE * stream = util_fopen(config_file , "r");
  int size = util_count_file_lines(stream);
  int ieq;
  equil_config_type * config = equil_config_alloc(size);
  for (ieq = 0; ieq < size; ieq++) {
    int elements_read = fscanf(stream , "%lg %lg %lg %lg %d %d %d" ,    
			       &config->datum_depth[ieq]  		   , 
			       &config->datum_P[ieq]      		   ,
			       &config->oil_water_Pc[ieq] 		   , 
			       &config->gas_oil_Pc[ieq]   		   ,           
			       &config->live_oil_init_mode[ieq]            ,
			       &config->black_oil_wgas_init_mode[ieq]      ,
			       &config->init_accuracy[ieq]);
    if (elements_read != 7) {
      fprintf(stderr,"%s: something wrong on line:%d in file: %s - aborting \n",__func__ , ieq + 1 , config_file);
      abort();
    }
    scalar_config_fscanf_line(config->scalar_config , ieq        , stream);  /* WOC */
    scalar_config_fscanf_line(config->scalar_config , ieq + size , stream);  /* GOC */
  }
  
  fclose(stream);
  return config;
}


int equil_config_get_nequil(const equil_config_type * equil_config) {
  return equil_config->data_size ;
}
    

void equil_config_free(equil_config_type * config) {
  free(config->datum_depth);
  free(config->datum_P);
  free(config->oil_water_Pc);
  free(config->gas_oil_Pc);
  free(config->live_oil_init_mode);
  free(config->black_oil_wgas_init_mode);
  free(config->init_accuracy);
  scalar_config_free(config->scalar_config);
  free(config);
}
							 

void equil_config_ecl_write(const equil_config_type * config  , const double * woc , const double * goc , FILE * stream) {
  int nequil = config->data_size;
  int ieq;
  fprintf(stream , "EQUIL\n");
  for (ieq=0; ieq < nequil; ieq++) 
    fprintf(stream , "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %5d  %5d  %5d /\n",
	    config->datum_depth[ieq]  		  , 
	    config->datum_P[ieq]      		  ,
	    woc[ieq]                   		  , 
	    config->oil_water_Pc[ieq] 		  , 
	    goc[ieq]           		          , 
            config->gas_oil_Pc[ieq]   		  ,           
            config->live_oil_init_mode[ieq]       ,
	    config->black_oil_wgas_init_mode[ieq] ,
	    config->init_accuracy[ieq]);
}


/*****************************************************************/

GET_DATA_SIZE(equil);
VOID_FUNC(equil_config_free , equil_config_type);


