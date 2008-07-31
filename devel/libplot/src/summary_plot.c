#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>
#include <plot_summary.h>
#include <ecl_kw.h>
#include <ecl_sum.h>

int main(int argc, const char **argv)
{
    plot_type *item;
    plot_dataset_type *d;
    double *x, *y;
    double x_max, y_max;
    int N;
    const char *kw = argv[1];

    item = plot_alloc();
    plot_set_window_size(item, 1152, 768);
    plot_initialize(item, "png", "punqs3.png", NORMAL);

    {
	char str[PATH_MAX];
	int i, k;
	int len = 2;
	int interval[len][2];

	interval[0][0] = 1;
	interval[0][1] = 10;
	interval[1][0] = 50;
	interval[1][1] = 60;

	for (i = 0; i < len; i++) {
	    for (k = interval[i][0]; k <= interval[i][1]; k++) {
		snprintf(str, PATH_MAX,
			 "/h/masar/EnKF_PUNQS3/PUNQS3/Original/Realizations/PUNQS3_Realization_%d/PUNQS3_%d.DATA",
			 k, k);
		plot_summary_collect_data(&x, &y, &N, str, kw);
		d = plot_dataset_alloc();
		plot_dataset_set_data(d, x, y, N, BLUE, LINE);
		plot_dataset_add(item, d);

		snprintf(str, PATH_MAX,
			 "/d/proj/bg/enkf/EnKF_PUNQS3/enkf_runs/member_%03d/PUNQS3_%04d.DATA",
			 k, k);
		plot_summary_collect_data(&x, &y, &N, str, kw);
		d = plot_dataset_alloc();
		plot_dataset_set_data(d, x, y, N, RED, LINE);
		plot_dataset_add(item, d);
	    }
	}
    }

    plot_summary_collect_data(&x, &y, &N,
			      "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			      kw);
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, POINT);
    plot_dataset_add(item, d);

    plot_set_labels(item, "Days", kw, "PUNQS3", BLACK);
    plot_get_maxima(item, &x_max, &y_max);
    plot_set_viewport(item, 0, x_max, 0, y_max);
    plot_data(item);
    plot_free(item);
}
