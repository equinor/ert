#ifndef __PLOT_DATASET_H__
#define __PLOT_DATASET_H__

typedef struct _plot_dataset {
     double     *xvalue; 
     double     *yvalue;
     int         length;
     plot_style  style;
     plot_color  color;
} plot_dataset;


extern plot_dataset *plot_dataset_alloc();
extern void          plot_dataset_free(plot_dataset *d);

extern void          plot_dataset_set_data(plot_dataset *d, double *x, double *y, int len, plot_color c, plot_style s);
extern int           plot_dataset_add(plot *item, plot_dataset *d);

#endif
