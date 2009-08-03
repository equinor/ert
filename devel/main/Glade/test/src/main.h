#ifndef __GUI_MAIN_H__
#define __GUI_MAIN_H__


#include <gtk/gtk.h>

#include "interface.h"
#include "support.h"

#include <enkf_main.h>
#include <model_config.h>
#include <ecl_config.h>
#include <path_fmt.h>
#include <util.h>
#include <ecl_grid.h>
#include <enkf_types.h>


#include <unistd.h>

enum
{
  TIMESTEP,
  ENSEMBLE_MEMBER,
  PROGRESS,
  PROGRESS_BUF,
  PROGRESS_ICON,
  COL_STATUS,
  ENKF_POINTER,
  N_COLS
};

enum
{
  SORT_TIMESTEP,
  SORT_ENSEMBLE_MEMBER
};

typedef enum _enkf_gui_run_node_type
{
  PARENT,
  CHILD
} enkf_gui_run_node;

typedef struct _enkf_gui_type
{
  enkf_gui_run_node type;
  gint ensemble_member;
  gint timestep;
} enkf_gui;


GtkTreeStore *enkf_gui_store_ensemble_member_parent ();
GtkTreeStore *enkf_gui_store_timestep_parent ();



#endif
