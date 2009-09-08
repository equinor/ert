#ifndef __PLOT_CONFIG_H__
#define __PLOT_CONFIG_H__
#include <config.h>

typedef struct plot_config_struct plot_config_type;


void               plot_config_set_path(plot_config_type * plot_config , const char * plot_path);
void               plot_config_set_image_type(plot_config_type * plot_config , const char * plot_device);
void               plot_config_set_viewer(plot_config_type * plot_config , const char * plot_viewer);
void               plot_config_set_driver(plot_config_type * plot_config , const char * plot_driver);;

const char  *      plot_config_get_path(const plot_config_type * plot_config );
const char  *      plot_config_get_image_type(const plot_config_type * plot_config );
const char  *      plot_config_get_viewer(const plot_config_type * plot_config );
const char  *      plot_config_get_driver(const plot_config_type * plot_config );
void               plot_config_free( plot_config_type * plot_config);
plot_config_type * plot_config_alloc();
void               plot_config_init_from_config(plot_config_type * plot_config , const config_type * config );

#endif
