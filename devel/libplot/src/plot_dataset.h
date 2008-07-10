#ifndef __PLOT_DATASET_H__
#define __PLOT_DATASET_H__

typedef struct plot_dataset_struct plot_dataset_type;
struct plot_dataset_struct {
     double          *xvalue;
     double          *yvalue;
     double           std_y;
     int              length;
     plot_style_type  style;
     plot_color_type  color;
};


extern plot_dataset_type *plot_dataset_alloc();
extern void               plot_dataset_free(plot_dataset_type *d);
extern void               plot_dataset_set_data(plot_dataset_type *d, double *x, double *y, int len, plot_color_type c, plot_style_type s);
extern int                plot_dataset_add(plot_type *item, plot_dataset_type *d);

#endif
