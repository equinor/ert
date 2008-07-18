#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>
#include <plot_canvas.h>
#include <glib.h>
#include <gtk/gtk.h>
#include <ecl_kw.h>
#include <ecl_sum.h>

/***************************************************************
 ***************************************************************/

typedef struct summary_plot_struct {
    plot_type *item;
    list_type *list;
} summary_plot_type;


typedef struct summary_plot_member_struct {
    char **summary_file_list;
    char *header_file;
    int files;
    const char *dir;
    const char *file;
    int last_report_step;
} summary_plot_member_type;

/***************************************************************
 ***************************************************************/

static summary_plot_type *summary_plot_alloc();
static void summary_plot_free(summary_plot_type * sp);
static summary_plot_member_type *summary_plot_member_alloc();
static void summary_plot_member_free(summary_plot_member_type * spm);
static void summary_plot_get_ecl_data(summary_plot_member_type * spm,
				      char ***sfl, char **header_file,
				      int *files);
static void summary_plot_add_ensamble_member(summary_plot_type * sp,
					     const char *dir,
					     const char *file);
static void destroy_local(GtkWidget * widget, gpointer data);
static gboolean summary_plot_timout(gpointer data);

/***************************************************************
 ***************************************************************/

summary_plot_type *summary_plot_alloc()
{
    summary_plot_type *sp;
    sp = malloc(sizeof *sp);
    sp->list = list_alloc();
    return sp;
}

void summary_plot_free(summary_plot_type * sp)
{
    list_node_type *node, *next_node;

    node = list_get_head(sp->list);
    while (node != NULL) {
	summary_plot_member_type *tmp;
	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
	list_del_node(sp->list, node);
	summary_plot_member_free(tmp);

	node = next_node;
    }

    list_free(sp->list);
    plot_free(sp->item);
    util_safe_free(sp);

}

summary_plot_member_type *summary_plot_member_alloc()
{
    summary_plot_member_type *spm;
    spm = malloc(sizeof *spm);
    spm->header_file = NULL;
    spm->last_report_step = 0;
    return spm;
}

void summary_plot_member_free(summary_plot_member_type * spm)
{
    util_free_stringlist(spm->summary_file_list, spm->files);
    util_safe_free(spm->header_file);
    util_safe_free(spm);
}

void summary_plot_get_ecl_data(summary_plot_member_type * spm, char ***sfl,
			       char **header_file, int *files)
{
    char data_file[PATH_MAX];
    char *path;
    char *base;
    char *header;
    bool fmt_file, unified;
    char **summary_file_list;
    int j;

    printf("Looking for new summary files in %s\n", spm->dir);
    snprintf(data_file, PATH_MAX, "%s/%s", spm->dir, spm->file);
    util_alloc_file_components(data_file, &path, &base, NULL);
    ecl_util_alloc_summary_files(path, base, &header,
				 &summary_file_list, &j,
				 &fmt_file, &unified);
    *sfl = summary_file_list;
    *header_file = header;
    *files = j;

    util_safe_free(path);
    util_safe_free(base);
}

void summary_plot_add_ensamble_member(summary_plot_type * sp,
				      const char *dir, const char *file)
{
    summary_plot_member_type *spm;

    spm = summary_plot_member_alloc();
    spm->dir = dir;
    spm->file = file;
    summary_plot_get_ecl_data(spm, &spm->summary_file_list,
			      &spm->header_file, &spm->files);
    list_append_ref(sp->list, spm);
}

gboolean summary_plot_timout(gpointer data)
{
    list_node_type *node, *next_node;
    summary_plot_type *sp = data;
    char **summary_file_list;
    char *header_file;
    int j, i, k, k2;
    ecl_sum_type *ecl_sum;
    int report_step, first_report_step, last_report_step;
    double *x, *y;
    double diff_day;
    time_t t, t0;


    node = list_get_head(sp->list);
    while (node != NULL) {
	summary_plot_member_type *tmp;
	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);

	summary_plot_get_ecl_data(tmp, &summary_file_list, &header_file,
				  &j);

	if (tmp->files != j || tmp->last_report_step == 0) {
	    util_free_stringlist(tmp->summary_file_list, tmp->files);
	    tmp->summary_file_list = summary_file_list;

	    ecl_sum = ecl_sum_fread_alloc(header_file, j,
					  (const char **) tmp->
					  summary_file_list, true, true);
	    ecl_sum_get_report_size(ecl_sum, &first_report_step,
				    &last_report_step);

	    x = malloc(sizeof(double) * last_report_step + 1);
	    y = malloc(sizeof(double) * last_report_step + 1);

	    for (report_step = first_report_step;
		 report_step <= last_report_step; report_step++) {
		if (ecl_sum_has_report_nr(ecl_sum, report_step)) {
		    int day, month, year;
		    util_set_date_values(ecl_sum_get_sim_time
					 (ecl_sum, report_step), &day,
					 &month, &year);
/*
		    printf("Report step %d: %d/%d/%d\n", report_step,
			   day, month, year);
*/
		    if (report_step == first_report_step)
			plot_util_get_time(day, month, year, &t0, NULL);
		    if (!t0) {
			fprintf(stderr,
				"!!!! Error: no first report step was found\n");
			continue;
		    }

		    plot_util_get_time(day, month, year, &t, NULL);
		    plot_util_get_diff(&diff_day, t, t0);

		    x[report_step] = (double) diff_day;
		    y[report_step] =
			ecl_sum_get_general_var(ecl_sum, report_step,
						"FOPT");
		}
	    }
	    /* If we have this is first time we want to plot up to the current step */
	    if (tmp->last_report_step == 0) {

		plplot_canvas_col0(plot_get_canvas(sp->item), BLACK);
		plplot_canvas_line(plot_get_canvas(sp->item),
				   last_report_step, x, y);
		plplot_canvas_adv(plot_get_canvas(sp->item), 0);

	    } else {
		/* Join lines between all the next points (steps) */
		for (i = 0; i < (j - tmp->files); i++) {
		    k = tmp->last_report_step + i - 1;
		    k2 = last_report_step - (j - tmp->files) + i;

		    printf("plotting from %d -> %d: %f, %f to %f, %f\n",
			   k, k2, x[k], y[k], x[k2], y[k2]);
		    plplot_canvas_join(plot_get_canvas(sp->item),
				       x[k], y[k], x[k2], y[k2]);
		}
	    }
	    tmp->last_report_step = last_report_step;
	    tmp->files = j;

	    ecl_sum_free(ecl_sum);
	} else {
	    util_free_stringlist(summary_file_list, j);
	    util_safe_free(header_file);
	}

	node = next_node;
    }

    return true;
}

void destroy_local(GtkWidget * widget, gpointer data)
{
    gtk_main_quit();
}

/***************************************************************
 ***************************************************************/

int main(int argc, char **argv)
{
    GtkWidget *win;
    PlplotCanvas *canvas;
    summary_plot_type *sp;
    summary_plot_type *sp_gurbat;

    gtk_init(&argc, &argv);

    sp = summary_plot_alloc();
    sp->item = plot_alloc();
    plot_initialize(sp->item, NULL, NULL, CANVAS);
    canvas = plot_get_canvas(sp->item);
    plot_set_viewport(sp->item, 0, 6000, 0, 3787500);

/*
    sp_gurbat = summary_plot_alloc();
    sp_gurbat->item = plot_alloc();
    plot_initialize(sp_gurbat->item, NULL, NULL, CANVAS);
    canvas = plot_get_canvas(sp_gurbat->item);
    plot_set_viewport(sp_gurbat->item, 0, 6000, 0, 29868498);
*/


    win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_container_set_border_width(GTK_CONTAINER(win), 0);
    g_signal_connect(G_OBJECT(win), "destroy", G_CALLBACK(destroy_local),
		     NULL);
    gtk_container_add(GTK_CONTAINER(win), GTK_WIDGET(canvas));

    g_timeout_add(5000, summary_plot_timout, sp);
    gtk_widget_show_all(win);
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_001",
				     "PUNQS3_0001.DATA");
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_002",
				     "PUNQS3_0002.DATA");
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_008",
				     "PUNQS3_0008.DATA");

    summary_plot_timout(sp);
    gtk_main();

    summary_plot_free(sp);

    return true;
}
