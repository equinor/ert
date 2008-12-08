#ifndef __PLOT_RANGE_H__
#define __PLOT_RANGE_H__
#ifdef __cplusplus
extern "C" {
#endif


typedef struct plot_range_struct plot_range_type;


plot_range_type     * plot_range_alloc();
void                  plot_range_free(plot_range_type *);


#ifdef __cplusplus
}
#endif
#endif

