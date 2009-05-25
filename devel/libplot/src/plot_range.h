#ifndef __PLOT_RANGE_H__
#define __PLOT_RANGE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>


typedef struct plot_range_struct plot_range_type;

typedef enum {
  MANUAL_RANGE =  1,      /* NO bitwise overlap with any of the other fields. */
  //AUTO_XMIN    =  2,      /* 2^n */
  //AUTO_XMAX    =  4,      /* 2^n */
  //AUTO_X       =  6,      /* auto_x = auto_xmin + auto_xmax */
  //AUTO_YMIN    =  8,      /* 2^n */
  //AUTO_YMAX    = 16,      /* 2^n */
  //AUTO_Y       = 24,      /* auto_y = auto_ymin + auto_ymax */ 
  AUTO_RANGE   = 30       /* auto_range == auto_x + auto_y */
} plot_range_mode_type;



plot_range_type     * plot_range_alloc();
void                  plot_range_free(plot_range_type *);

void plot_range_fprintf(const plot_range_type * , FILE * );

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

void plot_range_set_top_padding(plot_range_type    *  , double );
void plot_range_set_bottom_padding(plot_range_type *  , double );
void plot_range_set_left_padding(plot_range_type   *  , double );
void plot_range_set_right_padding(plot_range_type  *  , double );

void plot_range_invert_y_axis(plot_range_type * , bool );
void plot_range_invert_x_axis(plot_range_type * , bool );

void 		     plot_range_apply(plot_range_type * , double * , double * , double * , double *);
void 		     plot_range_set_manual_range( plot_range_type * range , double xmin , double xmax , double ymin , double ymax);
plot_range_mode_type plot_range_get_mode( const plot_range_type * range );

#ifdef __cplusplus
}
#endif
#endif

