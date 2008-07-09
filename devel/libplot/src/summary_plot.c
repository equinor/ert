#include "plot.h"
#include <ecl_kw.h>
#include <ecl_sum.h>

/*
 in directory: /d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original
 bash% /h/masar/EnKF/libecl/src/summary.x PUNQS3.DATA WOPR:PRO1 WOPR:PRO4 WOPR:PRO5
WOPR:PRO11 WOPR:PRO12 WOPR:PRO15

*/


static void collect_summary_data(double **x, double **y, int *size, const char *keyword) {
     const char data_file[] = "/d/proj/bg/enkf/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA";
     char *base;
     char *header_file;
     char **summary_file_list;
     char *path;
     int files;
     bool fmt_file, unified;
     ecl_sum_type *ecl_sum;
     int report_step, first_report_step, last_report_step;
     double *x_tmp, *y_tmp;

     util_alloc_file_components(data_file, &path, &base, NULL);

     ecl_util_alloc_summary_files(path, base, &header_file, &summary_file_list,
	   &files, &fmt_file, &unified);

     ecl_sum = ecl_sum_fread_alloc(header_file, files,
	   (const char **) summary_file_list, true, true );
     ecl_sum_get_report_size(ecl_sum , &first_report_step , &last_report_step);

     x_tmp = malloc(sizeof(double) * (files + 1));
     y_tmp = malloc(sizeof(double) * (files + 1));

     *size = files;

     for (report_step = first_report_step; report_step <= last_report_step; report_step++) {
	  if (ecl_sum_has_report_nr(ecl_sum , report_step)) {
	       int day, month, year;
	       util_set_date_values(ecl_sum_get_sim_time(ecl_sum , report_step) , &day , &month, &year);
/*	       fprintf(stdout, "%04d   %02d/%02d/%04d   ", report_step, day, month, year);*/

	       x_tmp[report_step] = (double) report_step; 
	       y_tmp[report_step] = ecl_sum_get_general_var(ecl_sum, report_step, keyword);
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



int main(int argc, const char **argv) {
     plot *p;
     plot_item *item;
     double *x, *y;
     double *y_tot = NULL;
     double *x_tot = NULL;
     int N, j;
     /* const char *keywords[] = {"WOPR:PRO1", "WOPR:PRO4"}; */
     const char *keywords[] = {"WOPR:PRO1", "WOPR:PRO4", "WOPR:PRO5", "WOPR:PRO11", "WOPR:PRO12", "WOPR:PRO15"};
     int nwords = 6;
     int i;

     p = plot_alloc();
     plot_init(p, true, argc, argv);

     item = plot_item_new(p, "png", "punqs3_wopr.png");
     for (j = 0; j < nwords; j++) {
	  collect_summary_data(&x, &y, &N, keywords[j]);
	  plot_item_set_graph_data(p, item, x, y, N);
     }
     plot_item_set_labels(item, "Timesteps", "WOPR:PRO1", "PUNQS3 test");
     plot_item_set_viewport(item, 0, 83, 0, 210);
     plot_item_plot_data(p, item, LINE);
     plot_item_free(p->plots, item);

     
     printf("--------------------------------------------\n");

     item = plot_item_new(p, "png", "punqs3_all_wopr.png");
     /* Calculate total production for all wells */ 
     for (j = 0; j < nwords; j++) {
	  collect_summary_data(&x, &y, &N, keywords[j]);
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
     
     plot_item_set_graph_data(p, item, x_tot, y_tot, N);
     plot_item_set_labels(item, "Timesteps", "WOPR, sum", "PUNQS3 test");
     plot_item_set_viewport(item, 0, 85, 0, 1200);
     plot_item_plot_data(p, item, LINE);
     plot_item_free(p->plots, item);

     plot_free(p);
}
