#include <enkf_types.h>
#include <util.h>
#include <enkf_macros.h>
#include <hash.h>
#include <ecl_grid.h>
#include <math.h>
#include <scalar_config.h>
#include <string.h>
#include <pilot_point_config.h>
#include <stringlist.h>


typedef double (variogram_func_type)  (double , double , double , double , double , double);


typedef struct  {
  variogram_func_type   * func;
  double                  x_range;
  double                  y_range;
  double                  z_range;
  double                  phase;    /* Radians */
  double                  std;
} variogram_type;


struct pilot_point_config_struct {
  scalar_config_type  * scalar_config;
  variogram_type      * variogram;
  const ecl_grid_type * grid;             /* Shared reference - not owned by the pilot_point_config instance. */
  int                 * index_list;       /* List of 1D indexes INTO THE GRID for the pilot points. */
  stringlist_type     * name_list;
  double              * kriging_weights;  /* IFF the variance structure does not change the kriging equations are solved ONCE. */
};


/*****************************************************************/

double exp_variogram(double x , double  y , double  z , double x_range , double y_range , double z_range) {
  return 1 - (exp(-x/x_range) * exp(-y/y_range) * exp(-z/z_range));
}

/*****************************************************************/

variogram_type * variogram_alloc(const char * _func_name , double x_range, double y_range , double z_range , double phase , double std) {
  variogram_type * variogram = util_malloc(sizeof * variogram , __func__);
  char * func_name = util_alloc_strupr_copy(_func_name);

  if (strcmp(func_name , "EXPONENTIAL") == 0)
    variogram->func = exp_variogram;
  else
    util_abort("%s: variogram function:%s not recognized \n",__func__ , _func_name);

  variogram->x_range = x_range;
  variogram->y_range = y_range;
  variogram->z_range = z_range;
  variogram->phase   = phase;
  variogram->std     = std;
  
  free(func_name);
  return variogram;
}


variogram_type * variogram_fscanf_alloc(FILE * stream) {
  char name[128];
  double   std , x_range , y_range , z_range , phase;
  if (fscanf(stream , "%s %lg %lg %lg %lg %lg" , name , &std , &x_range , &y_range , &z_range , &phase) != 6)
    util_abort("%s: failed to load variogram info \n",__func__);
  
  return variogram_alloc(name , x_range , y_range , z_range , phase , std);
}


 void variogram_free(variogram_type * variogram) {
  free( variogram );
}

/*****************************************************************/


static pilot_point_config_type * pilot_point_config_alloc_empty(int size, const ecl_grid_type * ecl_grid) {
  pilot_point_config_type * config  = util_malloc(sizeof * config , __func__);
  
  config->variogram     = NULL;
  config->scalar_config = scalar_config_alloc_empty( size );
  config->grid          = ecl_grid;
  config->index_list    = util_malloc(sizeof * config->index_list , __func__);
  config->name_list     = stringlist_alloc_new();
  return config;
}


/** 
    The confiog object takes ownership to the variogram.
*/
void pilot_point_config_set_variogram(pilot_point_config_type * config , variogram_type * variogram) {
  if (config->variogram != NULL)
    variogram_free(config->variogram);
  
  config->variogram = variogram;
}



/**
   The format of the file is as follows:

   1. The first line describes the variogram. It should look like this:

           variogram_type  Std   xrange yrange zrange angle
  
      Here variogram_type is the name of the variogram, it can be
      'EXPONENTIAL'. Std is standard deviation used in the variogram,
      xrange, yrange and zrange are three ranges and angle is the
      azimuth angle in the xy plane of the variogram.
 
   2. The following points are like this:

          NAME    i j k    UNIFORM 0 100
          NAME    i j k    GAUSS   0  25 
          ....

      I.e. one line for each pilot point; each line starts with a
      name, continues with a i,j,k value in the grid, and the
      distribution properties of this pilot point.
*/


pilot_point_config_type * pilot_point_config_fscanf_alloc(const char * config_file , const ecl_grid_type * ecl_grid) {
  pilot_point_config_type * config = NULL;
  variogram_type * variogram;
  FILE * stream = util_fopen(config_file , "r");
  int line_nr = 0;
  int size;
  
  size = util_count_file_lines(stream) - 1;
  fseek(stream , 0L , SEEK_SET);
  
  config    = pilot_point_config_alloc_empty(size , ecl_grid);
  variogram = variogram_fscanf_alloc( stream );
  pilot_point_config_set_variogram( config , variogram );
  do {
    char name[128];  /* UGGLY HARD CODED LIMIT */
    int  grid_index,i,j,k;
    if (fscanf(stream , "%s %d %d %d" , name , &i , &j , &k) != 4) 
      util_abort("%s: something wrong when reading: %s - aborting \n",__func__ , config_file);
    
    /* From ECLIPSE 1-offset to C 0-offset. */
    i -= 1;
    j -= 1;
    k -= 1;
    
    grid_index = ecl_grid_get_global_index(ecl_grid , i,j,k);
    stringlist_append_copy( config->name_list , name);
    scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
    line_nr++;
  } while ( line_nr < size );
  fclose(stream);
  
  return config;
}



void pilot_point_config_free(pilot_point_config_type * config) {
  variogram_free(config->variogram);
  scalar_config_free(config->scalar_config);
  stringlist_free(config->name_list);
  free(config->index_list);
  free(config);
}



scalar_config_type * pilot_point_config_get_scalar_config(const pilot_point_config_type * config) {
  return config->scalar_config;
}
