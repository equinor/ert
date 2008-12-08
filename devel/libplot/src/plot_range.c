#include <stdbool.h>
#include <stdlib.h>
#include <plot_range.h>
#include <util.h>

/**
   This file implements some simple functionality to manipulate / do
   book-keeping on the ranges of a plot. It is used in essentially two
   ways:
   
    1. As a storage area for recording the min and max of x and
       y. Both for automatic sets by inspecting the dataset, and for
       manual setting.

    2. The dataset have access to the range instance of the plot, so
       they can get min/max values when plotting. This is usefull when
       for instance plotting a line y = 5.

*/

typedef enum {
  manual_range = 1,
  auto_x       = 2,
  auto_y       = 4,
  auto_range   = 6       /* auto == auto_x + auto_y */
} plot_range_auto_type;


struct plot_range_struct {
  double 	        xmin     , xmax     , ymin     , ymax;      /* The actual min and max values. */
  bool                  xmin_set , xmax_set , ymin_set , ymax_set;  /* To ensure that all have been set to a valid value before use. */
  plot_range_auto_type  range_type;
};




plot_range_type * plot_range_alloc() {
  plot_range_type * range = util_malloc(sizeof * range , __func__);

  range->range_type = auto_range;
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



