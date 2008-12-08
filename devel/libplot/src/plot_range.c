#include <stdbool.h>
#include <stdlib.h>
#include <plplot/plplot.h>
#include <plplot/plplotP.h>
#include <plot_const.h>
#include <plot_range.h>
#include <util.h>
#include <arg_pack.h>

/**
   This file implements some simple functionality to manipulate / do
   book-keeping on the ranges of a plot. It is used in essentially two
   ways:
   
    1. As a storage area for recording the min and max of x and
       y. Both for automatic sets by inspecting the dataset, and for
       manual setting.

    2. The dataset have access to the range instance of the plot, so
       they can query for min/max values when plotting. This is
       usefull when for instance plotting a line y = 5.
*/


/** 
    There are four different values: min and max for x and y. All four
    can be individually controlled as manual or auto. Observe that the
    final range_mode should be considered as a mask. The relations
    documented in the enum below must be satisfied.
*/

typedef enum {
  manual_range =  1,      /* NO bitwise overlap with any of the other fields. */
  auto_xmin    =  2,      /* 2^n */
  auto_xmax    =  4,      /* 2^n */
  auto_x       =  6,      /* auto_x = auto_xmin + auto_xmax */
  auto_ymin    =  8,      /* 2^n */
  auto_ymax    = 16,      /* 2^n */
  auto_y       = 24,      /* auto_y = auto_ymin + auto_ymax */ 
  auto_range   = 30       /* auto_range == auto_x + auto_y */
} plot_range_mode_type;



struct plot_range_struct {
  double 	        xmin     , xmax     , ymin     , ymax;      /* The actual min and max values. */
  bool                  xmin_set , xmax_set , ymin_set , ymax_set;  /* To ensure that all have been set to a valid value before use. */
  plot_range_mode_type  range_mode;
};



static void plot_range_set_xmin__(plot_range_type * plot_range , double xmin) {
  plot_range->xmin     = xmin;
  plot_range->xmin_set = true;
}

static void plot_range_set_xmax__(plot_range_type * plot_range , double xmax) {
  plot_range->xmax     = xmax;
  plot_range->xmax_set = true;
}

static void plot_range_set_ymin__(plot_range_type * plot_range , double ymin) {
  plot_range->ymin     = ymin;
  plot_range->ymin_set = true;
}

static void plot_range_set_ymax__(plot_range_type * plot_range , double ymax) {
  plot_range->ymax     = ymax;
  plot_range->ymax_set = true;
}


/**
   Allocates a plot_range instance, and initializes it to the
   'auto_range' mode. If you want to use another mode you must call
   plot_range_set_mode() explicitly.
*/

plot_range_type * plot_range_alloc() {
  plot_range_type * range = util_malloc(sizeof * range , __func__);

  range->range_mode = auto_range;
  range->xmin       = 0;
  range->xmax       = 0;
  range->ymin       = 0;
  range->ymax       = 0;

  range->xmin_set   = false;
  range->xmax_set   = false;
  range->ymin_set   = false;
  range->ymax_set   = false;

  return range;
}



void plot_range_free(plot_range_type * plot_range) {
  free(plot_range);
}


/**
   This function sets the final range on the output device. Currently
   only PLPLOT.
*/

void plot_range_apply(plot_range_type * plot_range , arg_pack_type * plot_data) {
  if ((plot_range->xmin_set && plot_range->xmax_set) && (plot_range->ymin_set && plot_range->ymax_set)) {
    int stream = arg_pack_iget_int(plot_data , 0);
    plsstrm(stream);
    plvsta();  /* Sets up a standard viewport with padding ++ */
    plwind(plot_range->xmin, plot_range->xmax, plot_range->ymin, plot_range->ymax);

    plcol0(BLACK);
    plschr(0, LABEL_FONTSIZE);
    plbox("bcnst", 0.0 , 0 , "bcnstv" , 0.0 , 0);
  } else
    util_abort("%s: internal error - not all range values have been set: (%d,%d,%d,%d) \n",
	       plot_range->xmin_set,
	       plot_range->xmax_set,
	       plot_range->ymin_set,
	       plot_range->ymax_set);
}
