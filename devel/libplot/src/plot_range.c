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
  plot_range_mode_type  mode;
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
  range->mode = AUTO_RANGE;
  
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


void plot_range_set_manual_range( plot_range_type * range , double xmin , double xmax , double ymin , double ymax) {
  range->mode = MANUAL_RANGE;
  plot_range_set_xmin(range , xmin);
  plot_range_set_xmax(range , xmax);
  plot_range_set_ymin(range , ymin);
  plot_range_set_ymax(range , ymax);
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
  double x1 = 0;
  double x2 = 0;
  double y1 = 0;
  double y2 = 0;
  if (plot_range->mode == AUTO_RANGE) {
    double xmin 	= plot_range_get__(plot_range , XMIN );
    double xmax 	= plot_range_get__(plot_range , XMAX );
    double ymin 	= plot_range_get__(plot_range , YMIN );
    double ymax 	= plot_range_get__(plot_range , YMAX );
    double width  = fabs(xmax - xmin);
    double height = fabs(ymax - ymin);
    
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
  } else if (plot_range->mode == MANUAL_RANGE) {
    y1 = plot_range_get__(plot_range , YMIN );
    y2 = plot_range_get__(plot_range , YMAX );
    x1 = plot_range_get__(plot_range , XMIN );
    x2 = plot_range_get__(plot_range , XMAX );
  } else 
    util_exit("%s: sorry only MANUAL_RANGE and AUTO_RANGE implemented \n",__func__);

  
  /* Special case for only one point. */
  {
    if (x1 == x2) {
      if (x1 == 0) {
	x1 = -0.50;
	x2 =  0.50;
      } else {
	x1 -= 0.05 * abs(x1);
	x2 += 0.05 * abs(x2);
      }
    }
    
    if (y1 == y2) {
      if (y1 == 0.0) {
	y1 = -0.50;
	y2 =  0.50;
      } else {
	y1 -= 0.05 * abs(y1);
	y2 += 0.05 * abs(y2);
      }
    }
  }

  *_y1 = y1;
  *_y2 = y2;
  *_x1 = x1;
  *_x2 = x2;
}



plot_range_mode_type plot_range_get_mode( const plot_range_type * range ) {
  return range->mode;
}


