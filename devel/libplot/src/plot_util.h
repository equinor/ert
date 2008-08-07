#ifndef __PLOT_UTIL_H__
#define __PLOT_UTIL_H__
/**
 * @addtogroup plot_util_type plot_util_type: Different utilities that can come in handy.
 *
 * @{
 */


extern void plot_util_get_time(int mday, int mon, int year, time_t * t_ptr,
			       struct tm *time_ptr);
extern void plot_util_get_diff(double *mday, time_t t, time_t t0);
extern PLFLT plot_util_calc_rms(PLFLT * y, int len);

/**
 * @}
 */

#endif
