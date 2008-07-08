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

typedef struct _plot_dataset {
     double  *xvalue; 
     double  *yvalue;
     int      length;
} plot_dataset;

typedef struct __plot_item {
     list_type      *datasets; /* Different graph data in one plot */
     list_node_type *node;     /* points to the items node in the list */
     char           *filename;
     char           *device;

     char           *xlabel;
     char           *ylabel;
     char           *title;

     void            (*func)(void *data); /* data-manipulation function */
} plot_item;

typedef struct __plot_type {
     list_type *plots; /* linked list of different plots */
     bool       debug;
} plot_type;


/* plot.c */
extern plot_type       *plot_alloc();
extern void             plot_free(plot_type *p);
extern int              plot_init(plot_type *p, bool debug, int argc, const char **argv);


/* plot_item.c */
plot_item              *__plot_item_alloc();
plot_dataset           *__plot_dataset_alloc();
void                    __plot_item_free(list_type *list, plot_item *item);
extern plot_item       *plot_item_new(plot_type *p, char *dev, char *filename);
extern void             plot_item_set_graph_data(plot_type *p, plot_item *item, double *xvalue, double *yvalue, int length);
extern void             plot_item_set_labels(plot_item *item, char *xlabel, char *ylabel, char *title);
extern void             plot_item_set_size(plot_item *item, PLFLT xmin, PLFLT xmax, PLFLT ymin, PLFLT ymax);
extern void             plot_item_manipulate_data(plot_type *p, plot_item *item, void (*func)(void *data));
extern int              plot_item_plot_data(plot_type *p, plot_item *item);

#endif
