#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>

void plot_util_get_time(int mday, int mon, int year, time_t * t_ptr,
			struct tm *time_ptr)
{
    struct tm time;
    time_t t;

    time.tm_year = year - 1900;
    time.tm_mon = mon - 1;
    time.tm_mday = mday;
    time.tm_hour = 0;
    time.tm_min = 0;
    time.tm_sec = 1;
    time.tm_isdst = -1;
    t = mktime(&time);
    if (t == -1)
	(void) puts("-unknown-");

    if (time_ptr != NULL)
	*time_ptr = time;
    if (t_ptr != NULL)
	*t_ptr = t;
}

void plot_util_get_diff(PLFLT * mday, time_t t, time_t t0)
{
    PLFLT diff_sec, diff_min, diff_hour, diff_day;

    diff_sec = difftime(t, t0);
    diff_min = diff_sec / 60;
    diff_hour = diff_min / 60;
    diff_day = diff_hour / 24;

    /* Confirm with: http://www.onlineconversion.com/days_between.htm */
    *mday = diff_day;
}
