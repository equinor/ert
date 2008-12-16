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


#define XMIN 0
#define XMAX 1
#define YMIN 2
#define YMAX 3

struct plot_range_struct {
  double padding[4];
  double limits[4];
  double soft_limits[4];
  bool   set[4];
  bool   set_soft[4];
  bool   invert_x_axis;
  bool   invert_y_axis;
  plot_range_mode_type  range_mode;
};




static void plot_range_set__(plot_range_type * plot_range , int index , double value) {
  plot_range->limits[index] = value;
  plot_range->set[index] = true;
}

/*****************************************************************/

void plot_range_set_ymax(plot_range_type * plot_range , double ymax) {
  plot_range_set__(plot_range , YMAX , ymax);
}

void plot_range_set_ymin(plot_range_type * plot_range , double ymin) {
  plot_range_set__(plot_range , YMIN , ymin);
}

void plot_range_set_xmax(plot_range_type * plot_range , double xmax) {
  plot_range_set__(plot_range , XMAX , xmax);
}

void plot_range_set_xmin(plot_range_type * plot_range , double xmin) {
  plot_range_set__(plot_range , XMIN , xmin);
}

/*****************************************************************/

static void plot_range_set_soft__(plot_range_type * plot_range , int index , double value , bool lower_limit) {
  if (!plot_range->set_soft[index]) {
    plot_range->set_soft[index] = true;
    plot_range->soft_limits[index] = value;
  } else {
    if (lower_limit) {
      if (value < plot_range->soft_limits[index])
	plot_range->soft_limits[index] = value;
    } else {
      if (value > plot_range->soft_limits[index])
	plot_range->soft_limits[index] = value;
    }
  }
}


void plot_range_set_soft_ymax(plot_range_type * plot_range , double ymax) {
  plot_range_set_soft__(plot_range , YMAX , ymax , false);
}

void plot_range_set_soft_ymin(plot_range_type * plot_range , double ymin) {
  plot_range_set_soft__(plot_range , YMIN , ymin , true);
}

void plot_range_set_soft_xmax(plot_range_type * plot_range , double xmax) {
  plot_range_set_soft__(plot_range , XMAX , xmax , false);
}

void plot_range_set_soft_xmin(plot_range_type * plot_range , double xmin) {
  plot_range_set_soft__(plot_range , XMIN , xmin , true);
}


/*****************************************************************/
/* 
   These function will fail if the corresponding value has not
   been set, either from an automatic set, or manually.
*/

static double plot_range_safe_get__(const plot_range_type * plot_range , int index) {
  return plot_range->limits[index];
}

static double plot_range_get__(const plot_range_type * plot_range , int index) {
  if (plot_range->set[index])
    return plot_range_safe_get__(plot_range , index);
  else {
    util_abort("%s: tried to get xmin - but that has not been set.\n",__func__);
    return 0;
  }
}


double plot_range_get_xmin(const plot_range_type * plot_range) {
  return plot_range_get__(plot_range , XMIN);
}

double plot_range_get_xmax(const plot_range_type * plot_range) {
  return plot_range_get__(plot_range , XMAX);
}

double plot_range_get_ymin(const plot_range_type * plot_range) {
  return plot_range_get__(plot_range , YMIN);
}

double plot_range_get_ymax(const plot_range_type * plot_range) {
  return plot_range_get__(plot_range , YMAX);
}

/*****************************************************************/

double plot_range_safe_get_xmin(const plot_range_type * plot_range) {
  return plot_range_safe_get__(plot_range , XMIN);
}

double plot_range_safe_get_xmax(const plot_range_type * plot_range) {
  return plot_range_safe_get__(plot_range , XMAX);
}

double plot_range_safe_get_ymin(const plot_range_type * plot_range) {
  return plot_range_safe_get__(plot_range , YMIN);
}

double plot_range_safe_get_ymax(const plot_range_type * plot_range) {
  return plot_range_safe_get__(plot_range , YMAX);
}



/*****************************************************************/


/**
   Allocates a plot_range instance, and initializes it to the
   'auto_range' mode. If you want to use another mode you must call
   plot_range_set_mode() explicitly.
*/

plot_range_type * plot_range_alloc() {
  plot_range_type * range = util_malloc(sizeof * range , __func__);
  int i;
  range->range_mode = auto_range;

  for (i=0; i < 4; i++) {
    range->limits[i]      = 0;
    range->soft_limits[i] = 0;
    range->set[i]         = false;
    range->set_soft[i]    = false;
  }
  range->invert_x_axis = false;
  range->invert_y_axis = false;

  return range;
}



void plot_range_free(plot_range_type * plot_range) {
  free(plot_range);
}


static double plot_range_combined_get__(const plot_range_type * plot_range , int index , bool lower_limit) {
  
  if (plot_range->set_soft[index]) {
    if (lower_limit)
      return util_double_min(plot_range_get__(plot_range , index) , plot_range->soft_limits[index]);
    else
      return util_double_max(plot_range_get__(plot_range , index) , plot_range->soft_limits[index]);
  } else
    return plot_range_get__(plot_range , index);
  
}


void plot_range_apply(plot_range_type * plot_range) {
  double xmin = plot_range_combined_get__(plot_range , XMIN , true);
  double xmax = plot_range_combined_get__(plot_range , XMAX , false);
  double ymin = plot_range_combined_get__(plot_range , YMIN , true);
  double ymax = plot_range_combined_get__(plot_range , YMAX , false);

  double x1 , x2 , y1 , y2;

  if (plot_range->invert_x_axis) {
    x1 = xmax;
    x2 = xmin;
  } else {
    x1 = xmin;
    x2 = xmax;
  }

  if (plot_range->invert_y_axis) {
    y1 = ymax;
    y2 = ymin;
  } else {
    y1 = ymin;
    y2 = ymax;
  }
    
  plwind(x1,x2,y1,y2);
}
