#include <stdlib.h>
#include <util.h>
#include <plot_config.h>
#include <enkf_defaults.h>

/** 
    Struct holding basic information used when plotting.
*/

struct plot_config_struct {
  char * plot_path;     /* All the plots will be saved as xxxx files in this directory. */
  char * image_type;    /* Type of plot file - currently only 'png' is tested. */
  char * driver;        /* The driver used by the libplot layer when actually 'rendering' the plots. */
  char * viewer;        /* The executable used when displaying the newly created image. */
  int    height;   
  int    width;
};

/*****************************************************************/

void plot_config_set_width(plot_config_type * plot_config , int width) {
  plot_config->width = width;
}

void plot_config_set_height(plot_config_type * plot_config , int height) {
  plot_config->height = height;
}

void plot_config_set_path(plot_config_type * plot_config , const char * plot_path) {
  plot_config->plot_path = util_realloc_string_copy(plot_config->plot_path , plot_path);
  util_make_path( plot_path );
}


void plot_config_set_image_type(plot_config_type * plot_config , const char * image_type) {
  plot_config->image_type = util_realloc_string_copy(plot_config->image_type , image_type);
}


void plot_config_set_viewer(plot_config_type * plot_config , const char * plot_viewer) {
  plot_config->viewer = util_realloc_string_copy(plot_config->viewer , plot_viewer);
}


void plot_config_set_driver(plot_config_type * plot_config , const char * plot_driver) {
  plot_config->driver = util_realloc_string_copy(plot_config->driver , plot_driver);
}


/*****************************************************************/

const char *  plot_config_get_path(const plot_config_type * plot_config ) {
  return plot_config->plot_path;
}


const char *  plot_config_get_image_type(const plot_config_type * plot_config ) {
  return plot_config->image_type;
}

const char *  plot_config_get_viewer(const plot_config_type * plot_config ) {
  return plot_config->viewer;
}  

const char *  plot_config_get_driver(const plot_config_type * plot_config ) {
  return plot_config->driver;
}

int plot_config_get_width(const plot_config_type * plot_config ) {
  return plot_config->width;
}

int plot_config_get_height(const plot_config_type * plot_config ) {
  return plot_config->height;
}




void plot_config_free( plot_config_type * plot_config) {
  free(plot_config->plot_path);
  free(plot_config->viewer);
  free(plot_config->image_type);
  free(plot_config->driver );
  free(plot_config);
}


/**
   The plot_config object is instantiated with the default values from enkf_defaults.h
*/
plot_config_type * plot_config_alloc() {
  plot_config_type * info = util_malloc( sizeof * info , __func__);
  info->plot_path   = NULL;
  info->image_type  = NULL;
  info->viewer      = NULL;
  info->driver      = NULL;
  
  plot_config_set_path(info       , DEFAULT_PLOT_PATH );
  plot_config_set_image_type(info , DEFAULT_IMAGE_TYPE );
  plot_config_set_viewer(info     , DEFAULT_IMAGE_VIEWER );
  plot_config_set_driver(info     , DEFAULT_PLOT_DRIVER );
  plot_config_set_width(info      , DEFAULT_PLOT_WIDTH );
  plot_config_set_height(info     , DEFAULT_PLOT_HEIGHT );
  
  return info;
}



void plot_config_init_from_config(plot_config_type * plot_config , const config_type * config ) {
  if (config_item_set( config , "PLOT_PATH"))
    plot_config_set_path( plot_config , config_get_value( config , "PLOT_PATH" ));

  if (config_item_set( config , "PLOT_DRIVER"))
    plot_config_set_driver( plot_config , config_get_value( config , "PLOT_DRIVER" ));
  
  if (config_item_set( config , "IMAGE_VIEWER"))
    plot_config_set_viewer( plot_config , config_get_value( config , "IMAGE_VIEWER" ));

  if (config_item_set( config , "PLOT_DRIVER"))
    plot_config_set_driver( plot_config , config_get_value( config , "PLOT_DRIVER" ));
}

