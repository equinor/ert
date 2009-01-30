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
  bool   set[4];
  bool   invert_x_axis;
  bool   invert_y_axis;
  plot_range_mode_type  range_mode;
};

/*****************************************************************/

void plot_range_fprintf(const plot_range_type * range, FILE * stream) {
  printf("x1: %g    x2:%g   y1:%g   y2:%g \n",range->limits[XMIN] , range->limits[XMAX] , range->limits[YMIN] , range->limits[YMAX]);
}


/*****************************************************************/

static void plot_range_set__(plot_range_type * plot_range , int index , double value) {
  plot_range->limits[index] = value;
  plot_range->set[index] = true;
}

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
/* 
   These functions will fail if the corresponding value has not
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

static void plot_range_set_padding__(plot_range_type * plot_range , int index , double value) {
  plot_range->padding[index] = value;
}


void plot_range_set_left_padding(plot_range_type * plot_range , double value) {
  plot_range_set_padding__(plot_range , XMIN , value);
}

void plot_range_set_right_padding(plot_range_type * plot_range , double value) {
  plot_range_set_padding__(plot_range , XMAX , value);
}

void plot_range_set_top_padding(plot_range_type * plot_range , double value) {
  plot_range_set_padding__(plot_range , YMAX , value);
}

void plot_range_set_bottom_padding(plot_range_type * plot_range , double value) {
  plot_range_set_padding__(plot_range , YMIN , value);
}

/*****************************************************************/

void plot_range_invert_x_axis(plot_range_type * range, bool invert) {
  range->invert_x_axis = invert;
}

void plot_range_invert_y_axis(plot_range_type * range, bool invert) {
  range->invert_y_axis = invert;
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
    range->padding[i]     = 0;
    range->set[i]         = false;
  }

  range->invert_x_axis = false;
  range->invert_y_axis = false;
  return range;
}



void plot_range_free(plot_range_type * plot_range) {
  free(plot_range);
}



/**
   This function return the final xmin,xmax,ymin and ymax
   functions. To avvoid filling up plplot specific function calls,
   this function does not call plwind(), which would have been
   natural.

   From the calling scope:
   {
      double x1,x2,y1,y2;
      plot_range_apply(range , &x1 , &x2 , &y1 , &y2);
      plwind( x1,x2,y1,y2);
   }

*/

   
void plot_range_apply(plot_range_type * plot_range, double *_x1 , double *_x2 , double *_y1 , double *_y2) {
  double xmin 	= plot_range_get__(plot_range , XMIN );
  double xmax 	= plot_range_get__(plot_range , XMAX );
  double ymin 	= plot_range_get__(plot_range , YMIN );
  double ymax 	= plot_range_get__(plot_range , YMAX );
  double width  = fabs(xmax - xmin);
  double height = fabs(ymax - ymin);

  double x1 , x2 , y1 , y2;

  if (plot_range->invert_x_axis) {
    x1 = xmax;
    x2 = xmin;

    x1 += width * plot_range->padding[XMAX];
    x2 -= width * plot_range->padding[XMIN];
  } else {
    x1 = xmin;
    x2 = xmax;

    x1 -= width * plot_range->padding[XMIN];
    x2 += width * plot_range->padding[XMAX];
  }

  if (plot_range->invert_y_axis) {
    y1 = ymax;
    y2 = ymin;

    y1 += height * plot_range->padding[YMAX];
    y2 -= height * plot_range->padding[YMIN];
  } else {
    y1 = ymin;
    y2 = ymax;

    y1 -= height * plot_range->padding[YMIN];
    y2 += height * plot_range->padding[YMAX];
  }
  
  if (_y1 != NULL) *_y1 = y1;
  if (_y2 != NULL) *_y2 = y2;
  if (_x1 != NULL) *_x1 = x1;
  if (_x2 != NULL) *_x2 = x2;
}



