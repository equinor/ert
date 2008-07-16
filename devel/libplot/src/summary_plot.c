#include <plot.h>
#include <plot_dataset.h>
#include <ecl_kw.h>
#include <ecl_sum.h>

/*
 * in directory: /d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original bash%
 * /h/masar/EnKF/libecl/src/summary.x PUNQS3.DATA WOPR:PRO1 WOPR:PRO4
 * WOPR:PRO5 WOPR:PRO11 WOPR:PRO12 WOPR:PRO15
 * 
 */

static void
collect_summary_data(double **x, double **y, int *size,
		     const char *data_file, const char *keyword)
{
    char *base;
    char *header_file;
    char **summary_file_list;
    char *path;
    int files;
    bool fmt_file, unified;
    ecl_sum_type *ecl_sum;
    int report_step, first_report_step, last_report_step;
    double *x_tmp, *y_tmp;
    time_t t, t0;
    struct tm time;
    char buf[32];
    char timebuf[20];
    double diff_sec, diff_min, diff_hour, diff_day;
    struct tm time0;

    util_alloc_file_components(data_file, &path, &base, NULL);

    ecl_util_alloc_summary_files(path, base, &header_file,
				 &summary_file_list, &files, &fmt_file,
				 &unified);

    ecl_sum = ecl_sum_fread_alloc(header_file, files,
				  (const char **) summary_file_list, true,
				  true);
    ecl_sum_get_report_size(ecl_sum, &first_report_step,
			    &last_report_step);

    x_tmp = malloc(sizeof(double) * (files + 1));
    y_tmp = malloc(sizeof(double) * (files + 1));

    *size = files;

    /* 1 Feb 1967 */
    /* Grab this from the first element in the list */
    time0.tm_year = 1967 - 1900;
    time0.tm_mon = 1 - 1;
    time0.tm_mday = 1;
    time0.tm_hour = 0;
    time0.tm_min = 0;
    time0.tm_sec = 1;
    t0 = mktime(&time0);
    if (t0 == -1)
	(void) puts("-unknown-");

    for (report_step = first_report_step; report_step <= last_report_step;
	 report_step++) {
	if (ecl_sum_has_report_nr(ecl_sum, report_step)) {
	    int day, month, year;

	    util_set_date_values(ecl_sum_get_sim_time
				 (ecl_sum, report_step), &day, &month,
				 &year);
	    snprintf(buf, sizeof(buf), "%02d/%02d/%04d", day, month, year);
	    //util_sscanf_date(buf, &t);
	    //time = gmtime(&t);
	    time.tm_year = year - 1900;
	    time.tm_mon = month - 1;
	    time.tm_mday = day;
	    time.tm_hour = 0;
	    time.tm_min = 0;
	    time.tm_sec = 1;
	    time.tm_isdst = -1;
	    t = mktime(&time);
	    if (t == -1)
		(void) puts("-unknown-");
	    else {
		strftime(timebuf, sizeof(timebuf), "%d/%m/%Y %A", &time);
	    }

	    diff_sec = difftime(t, t0);
	    diff_min = diff_sec / 60;
	    diff_hour = diff_min / 60;
	    diff_day = diff_hour / 24;
	    /* Confirm with http://www.onlineconversion.com/days_between.htm 
	       printf("Days since 01/01/1967: %f\n", diff_day);
	     */

	    x_tmp[report_step] = (double) diff_day;
	    y_tmp[report_step] =
		ecl_sum_get_general_var(ecl_sum, report_step, keyword);
	}
    }
    *x = x_tmp;
    *y = y_tmp;

    util_safe_free(header_file);
    util_safe_free(base);
    util_safe_free(path);
    util_free_stringlist(summary_file_list, files);
    ecl_sum_free(ecl_sum);
}

/**************************************************************/
/**************************************************************/

int main(int argc, const char **argv)
{
    plot_type *item;
    plot_dataset_type *d;
    double *x, *y;
    double *y_tot = NULL;
    double *x_tot = NULL;
    int N, j;

    /*
     * const char *keywords[] = {"WOPR:PRO1", "WOPR:PRO4"}; 
     */
    const char *keywords[] =
	{ "WOPR:PRO1", "WOPR:PRO4", "WOPR:PRO5", "WOPR:PRO11",
	"WOPR:PRO12", "WOPR:PRO15"
    };
    int nwords = 6;
    int i;

    plparseopts(&argc, argv, PL_PARSE_FULL);

    item = plot_alloc();
    plot_initialize(item, "png", "punqs3_wopr.png");
    for (j = 0; j < nwords; j++) {
	collect_summary_data(&x, &y, &N,
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
    plot_initialize(item, "png", "punqs3_all_wopr.png");

    /*
     * Calculate total production for all wells 
     */
    for (j = 0; j < nwords; j++) {
	collect_summary_data(&x, &y, &N,
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
    plot_initialize(item, "png", "punqs3_fopt.png");

    {
	char str[PATH_MAX];
	int i;

	/* Add EnKF results.
	 * This data ran trough eclipse with 1 aquifer!
	 */
	for (i = 1; i <= 100; i += 20) {
	    snprintf(str, PATH_MAX,
		     "/d/proj/bg/enkf/PUNQS3_ORIG_RELMIN/tmp_%04d/PUNQS3_%04d.DATA",
		     i, i);
	    collect_summary_data(&x, &y, &N, str, "FOPT");
	    d = plot_dataset_alloc();
	    plot_dataset_set_data(d, x, y, N, RED, LINE);
	    plot_dataset_add(item, d);
	}
	/* Add RMS results */
	for (i = 1; i <= 100; i += 20) {
	    snprintf(str, PATH_MAX,
		     "/h/masar/EnKF_PUNQS3/PUNQS3/Original/Realizations/PUNQS3_Realization_%d/PUNQS3_%d.DATA",
		     i, i);
	    collect_summary_data(&x, &y, &N, str, "FOPT");
	    d = plot_dataset_alloc();
	    plot_dataset_set_data(d, x, y, N, BLUE, LINE);
	    plot_dataset_add(item, d);
	}
    }
    collect_summary_data(&x, &y, &N,
			 "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			 "FOPT");
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, POINT);
    plot_dataset_add(item, d);

    plot_set_labels(item, "Days", "FOPT", "PUNQS3 FOPT Original", BLACK);
    plot_set_viewport(item, 0, 6025, 0, 3873400);
    plot_data(item);
    plot_free(item);

}
