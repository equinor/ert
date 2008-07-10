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

typedef enum plot_style_enum {
    HISTOGRAM = 0,
    LINE = 1,
    POINT = 2
} plot_style_type;

typedef enum plot_color_enum {
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
} plot_color_type;

typedef struct plot_struct plot_type;
struct plot_struct {
    list_type *datasets;
    char *filename;
    char *device;

    char *xlabel;
    char *ylabel;
    char *title;
    plot_color_type label_color;
};

extern plot_type *plot_alloc();
extern void plot_initialize(plot_type * item, char *dev, char *filename);

extern void plot_set_labels(plot_type * item, char *xlabel,
			    char *ylabel, char *title,
			    plot_color_type color);
extern void plot_set_viewport(plot_type * item, PLFLT xmin, PLFLT xmax,
			      PLFLT ymin, PLFLT ymax);
extern void plot_data(plot_type * item);

extern void plot_free_all_datasets(plot_type * item);
extern void plot_free(plot_type * item);

#endif
