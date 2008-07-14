#ifndef __PLOT_DATASET_H__
#define __PLOT_DATASET_H__

typedef struct plot_dataset_struct plot_dataset_type;

/* Get functions */
extern int plot_datset_get_length(plot_dataset_type * d);
extern plot_color_type plot_datset_get_color(plot_dataset_type * d);
extern plot_style_type plot_datset_get_style(plot_dataset_type * d);
double *plot_datset_get_vector_x(plot_dataset_type * d);
double *plot_datset_get_vector_y(plot_dataset_type * d);

/* Set functions */
extern void plot_datset_set_style(plot_dataset_type * d,
				  plot_style_type s);


extern plot_dataset_type *plot_dataset_alloc();
extern void plot_dataset_free(plot_dataset_type * d);
extern void plot_dataset_set_data(plot_dataset_type * d, double *x,
				  double *y, int len,
				  plot_color_type c, plot_style_type s);
extern int plot_dataset_add(plot_type * item, plot_dataset_type * d);

#endif
