#ifndef __ENKF_SETUP_H__
#define __ENKF_SETUP_H__


#include <gtk/gtk.h>
#include <enkf_main.h>
#include <model_config.h>
#include <ecl_config.h>
#include <path_fmt.h>
#include <util.h>
#include <ecl_grid.h>
#include <enkf_types.h>
#include <sched_file.h>
#include <stringlist.h>
#include <stdlib.h>

enkf_main_type *enkf_setup_bootstrap(const char *enkf_config,
				     GtkWidget * win);


#endif
