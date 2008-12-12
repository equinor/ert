#ifndef __PLOT_RANGE_H__
#define __PLOT_RANGE_H__
#ifdef __cplusplus
extern "C" {
#endif


typedef struct plot_range_struct plot_range_type;


plot_range_type     * plot_range_alloc();
void                  plot_range_free(plot_range_type *);


double plot_range_get_xmax(const plot_range_type * );
double plot_range_get_ymax(const plot_range_type * );
double plot_range_get_xmin(const plot_range_type * );
double plot_range_get_ymin(const plot_range_type * );

#ifdef __cplusplus
}
#endif
#endif

