#ifndef __PLOT_H__
#define __PLOT_H__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <util.h>
#include <list.h>
#include <list_node.h>
#include <plplot/plplot.h>
#include <plplot/plplotP.h>

typedef enum __plot_style {
     HISTOGRAM = 0,
     LINE = 1,
     POINT = 2
} plot_style;

typedef enum __plot_color {
     WHITE = 0,
     RED = 1,
     YELLOW = 2,
     GREEN = 3,
     AQUAMARINE = 4,
     PINK = 5,
     WHEAT = 6,
     GRAY = 7,
     BROWN = 8,
     BLUE = 9,
     VIOLET = 10,
     CYAN = 11,
     TURQUOISE = 12,
     MAGENTA = 13,
     SALMON = 14,
     BLACK = 15
} plot_color;


typedef struct __plot {
     list_type      *datasets; 
     char           *filename;
     char           *device;

     char           *xlabel;
     char           *ylabel;
     char           *title;
     plot_color      label_color;
} plot;


extern plot *plot_alloc();
extern void  plot_initialize(plot *item, char *dev, char *filename);

extern void  plot_set_labels(plot *item, char *xlabel, char *ylabel, char *title, plot_color color);
extern void  plot_set_viewport(plot *item, PLFLT xmin, PLFLT xmax, PLFLT ymin, PLFLT ymax);
extern void  plot_data(plot *item);

extern void  plot_free_all_datasets(plot *item); 
extern void  plot_free(plot *item);

#endif
