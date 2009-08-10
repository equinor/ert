#ifndef __PLOT_RANGE_H__
#define __PLOT_RANGE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>

typedef struct plot_range_struct plot_range_type;



plot_range_type     * plot_range_alloc();
void                  plot_range_free(plot_range_type *);

void plot_range_fprintf(const plot_range_type * , FILE * );

double plot_range_get_final_xmin(const plot_range_type * plot_range);
double plot_range_get_final_xmax(const plot_range_type * plot_range);
double plot_range_get_final_ymin(const plot_range_type * plot_range);
double plot_range_get_final_ymax(const plot_range_type * plot_range);

double plot_range_get_xmax(const plot_range_type * );
double plot_range_get_ymax(const plot_range_type * );
double plot_range_get_xmin(const plot_range_type * );
double plot_range_get_ymin(const plot_range_type * );

double plot_range_safe_get_xmax(const plot_range_type * );
double plot_range_safe_get_ymax(const plot_range_type * );
double plot_range_safe_get_xmin(const plot_range_type * );
double plot_range_safe_get_ymin(const plot_range_type * );
 
void plot_range_set_xmax(plot_range_type *  , double);
void plot_range_set_ymax(plot_range_type *  , double);
void plot_range_set_xmin(plot_range_type *  , double);
void plot_range_set_ymin(plot_range_type *  , double);

void plot_range_set_auto_xmax(plot_range_type *  , double);
void plot_range_set_auto_ymax(plot_range_type *  , double);
void plot_range_set_auto_xmin(plot_range_type *  , double);
void plot_range_set_auto_ymin(plot_range_type *  , double);

void plot_range_set_top_padding(plot_range_type    *  , double );
void plot_range_set_bottom_padding(plot_range_type *  , double );
void plot_range_set_left_padding(plot_range_type   *  , double );
void plot_range_set_right_padding(plot_range_type  *  , double );

void plot_range_invert_y_axis(plot_range_type * , bool );
void plot_range_invert_x_axis(plot_range_type * , bool );

  void 		     plot_range_apply(plot_range_type * );
void 		     plot_range_set_range( plot_range_type * range , double xmin , double xmax , double ymin , double ymax);

#ifdef __cplusplus
}
#endif
#endif

