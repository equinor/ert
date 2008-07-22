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
    double *y_tot = NULL;
    double *x_tot = NULL;
    double x_max, y_max;
    int N, j;

    const char *keywords[] =
	{ "WOPR:PRO1", "WOPR:PRO4", "WOPR:PRO5", "WOPR:PRO11",
	"WOPR:PRO12", "WOPR:PRO15"
    };
    int nwords = 6;
    int i;

    plparseopts(&argc, argv, PL_PARSE_FULL);

    item = plot_alloc();
    plot_initialize(item, "png", "punqs3_wopr.png", NORMAL);
    for (j = 0; j < nwords; j++) {
	plot_summary_collect_data(&x, &y, &N,
			     "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			     keywords[j]);
	d = plot_dataset_alloc();
	plot_dataset_set_data(d, x, y, N, BROWN, LINE);
	plot_dataset_add(item, d);
    }
    plot_set_labels(item, "Timesteps", "WOPR:PRO1", "PUNQS3 test", BROWN);
    plot_set_viewport(item, 0, 6025, 0, 210);
    plot_data(item);
    plot_free(item);

    printf("--------------------------------------------\n");

    item = plot_alloc();
    plot_initialize(item, "png", "punqs3_all_wopr.png", NORMAL);

    /*
     * Calculate total production for all wells 
     */
    for (j = 0; j < nwords; j++) {
	plot_summary_collect_data(&x, &y, &N,
			     "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			     keywords[j]);
	if (!y_tot && !x_tot) {
	    x_tot = malloc(sizeof(double) * (N + 1));
	    y_tot = malloc(sizeof(double) * (N + 1));
	    memset(x_tot, 0, sizeof(double) * (N + 1));
	    memset(y_tot, 0, sizeof(double) * (N + 1));
	}
	for (i = 0; i <= N; i++) {
	    y_tot[i] = y_tot[i] + y[i];
	    x_tot[i] = x[i];
	}
	util_safe_free(y);
	util_safe_free(x);
    }

    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x_tot, y_tot, N, BROWN, LINE);
    plot_dataset_add(item, d);

    plot_set_labels(item, "Timesteps", "WOPR, sum", "PUNQS3 test", BROWN);
    plot_set_viewport(item, 0, 6025, 0, 1200);
    plot_data(item);
    plot_free(item);

    printf("--------------------------------------------\n");

    item = plot_alloc();
    plot_initialize(item, "png", "punqs3_fopt.png", NORMAL);

    {
	char str[PATH_MAX];
	int i;

	/* Add EnKF results.
	 * This data ran trough eclipse with 1 aquifer!
	 */
	for (i = 1; i <= 100; i += 20) {
	    snprintf(str, PATH_MAX,
		     "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3_ORIG_RELMIN/tmp_%04d/PUNQS3_%04d.DATA",
		     i, i);
	    plot_summary_collect_data(&x, &y, &N, str, "FOPT");
	    d = plot_dataset_alloc();
	    plot_dataset_set_data(d, x, y, N, RED, LINE);
	    plot_dataset_add(item, d);
	}
	/* Add RMS results */
	for (i = 1; i <= 100; i += 20) {
	    snprintf(str, PATH_MAX,
		     "/h/masar/EnKF_PUNQS3/PUNQS3/Original/Realizations/PUNQS3_Realization_%d/PUNQS3_%d.DATA",
		     i, i);
	    plot_summary_collect_data(&x, &y, &N, str, "FOPT");
	    d = plot_dataset_alloc();
	    plot_dataset_set_data(d, x, y, N, BLUE, LINE);
	    plot_dataset_add(item, d);
	}
    }
    plot_summary_collect_data(&x, &y, &N,
			 "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			 "FOPT");
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, POINT);
    plot_dataset_add(item, d);

    plot_set_labels(item, "Days", "FOPT", "PUNQS3 FOPT Original", BLACK);
    plot_util_get_maxima(item, &x_max, &y_max);
    plot_set_viewport(item, 0, x_max, 0, y_max);
    plot_data(item);
    plot_free(item);

}
