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

void plot_util_get_diff(double *mday, time_t t, time_t t0)
{
    double diff_sec, diff_min, diff_hour, diff_day;

    diff_sec = difftime(t, t0);
    diff_min = diff_sec / 60;
    diff_hour = diff_min / 60;
    diff_day = diff_hour / 24;

    /* Confirm with: http://www.onlineconversion.com/days_between.htm */
    *mday = diff_day;
}

void plot_util_get_maxima(plot_type * item, double *x_max, double *y_max)
{
    list_node_type *node, *next_node;
    double tmp_x = 0;
    double tmp_y = 0;
    double *x, *y;
    int i;

    node = list_get_head(plot_get_datasets(item));
    while (node != NULL) {
	plot_dataset_type *tmp;
	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
        x = plot_datset_get_vector_x(tmp);
        y = plot_datset_get_vector_y(tmp);

        for (i = 0; i <= plot_datset_get_length(tmp); i++) {
            if (x[i] > tmp_x)
                tmp_x = x[i];
            if (y[i] > tmp_y)
                tmp_y = y[i];
        }
	node = next_node;
    }
    
    fprintf(stderr, "ID[%d] Found maxima: x: %f and y: %f\n", plot_get_stream(item), tmp_x, tmp_y);
    *x_max = tmp_x;
    *y_max = tmp_y;
}
