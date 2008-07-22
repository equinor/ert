#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>
#include <plot_summary.h>
#include <glib.h>
#include <gtk/gtk.h>
#include <ecl_kw.h>
#include <ecl_sum.h>
#include <time.h>

/***************************************************************
 ***************************************************************/

typedef struct summary_plot_struct {
    plot_type *item;
    GtkTextBuffer *buffer;
    GtkWidget *text;
    list_type *list;
} summary_plot_type;


typedef struct summary_plot_member_struct {
    int files;
    char *dir;
    char *file;
    int last_report_step;
    char *keyword;
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
					     char *dir, char *file,
					     char *keyword);
static void summary_plot_append_textbox(summary_plot_type * sp,
					const char *str, ...);
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
    spm->last_report_step = 0;
    return spm;
}

void summary_plot_member_free(summary_plot_member_type * spm)
{
    util_safe_free(spm->dir);
    util_safe_free(spm->file);
    util_safe_free(spm->keyword);
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
    if (sfl)
	*sfl = summary_file_list;
    else
	util_free_stringlist(summary_file_list, j);

    if (header_file)
	*header_file = header;
    else
	util_safe_free(header);

    *files = j;
    util_safe_free(path);
    util_safe_free(base);
}

void summary_plot_add_ensamble_member(summary_plot_type * sp,
				      char *dir, char *file, char *keyword)
{
    summary_plot_member_type *spm;

    spm = summary_plot_member_alloc();
    spm->dir = strdup(dir);
    spm->file = strdup(file);
    spm->keyword = strdup(keyword);

    summary_plot_get_ecl_data(spm, NULL, NULL, &spm->files);
    list_append_ref(sp->list, spm);
}

gboolean summary_plot_timout(gpointer data)
{
    list_node_type *node, *next_node;
    summary_plot_type *sp = data;
    char **summary_file_list;
    char *header_file;
    int j;
    ecl_sum_type *ecl_sum;
    int report_step, first_report_step, last_report_step;
    double *x, *y;
    double diff_day;
    time_t t, t0;
    plot_dataset_type *d;


    node = list_get_head(sp->list);
    while (node != NULL) {
	summary_plot_member_type *tmp;
	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);

	summary_plot_get_ecl_data(tmp, &summary_file_list, &header_file,
				  &j);

	if (tmp->files != j || tmp->last_report_step == 0) {
	    ecl_sum = ecl_sum_fread_alloc(header_file, j, (const char **)
					  summary_file_list, true, true);
	    ecl_sum_get_report_size(ecl_sum, &first_report_step,
				    &last_report_step);

	    x = malloc(sizeof(double) * (last_report_step + 1));
	    y = malloc(sizeof(double) * (last_report_step + 1));

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
						tmp->keyword);
		}
	    }
	    /* If we have this is first time we want to plot up to the current step */
	    if (tmp->last_report_step == 0) {
		d = plot_dataset_alloc();
		plot_dataset_set_data(d, x, y, last_report_step, RED,
				      LINE);
		plot_dataset(sp->item, d);
		plot_dataset_free(d);

	    } else {
		/* Join lines between all the next points (steps) */
		d = plot_dataset_alloc();
		plot_dataset_set_data(d, x, y, last_report_step, RED,
				      LINE);
		plot_dataset_join(sp->item, d, tmp->last_report_step,
				  last_report_step);
		plot_dataset_free(d);
	    }
	    tmp->last_report_step = last_report_step;
	    tmp->files = j;

	    ecl_sum_free(ecl_sum);
	}

	util_free_stringlist(summary_file_list, j);
	util_safe_free(header_file);

	node = next_node;
    }

    return true;
}

char *summary_plot_get_timestamp()
{
    struct tm *ptr;
    time_t tm;
    char str[10];

    tm = time(NULL);
    ptr = localtime(&tm);
    strftime(str, sizeof(str), "%T",ptr);

    return strdup(str);
}

void summary_plot_append_textbox(summary_plot_type * sp, const char *str, ...)
{
    GtkTextIter iter;
    char buf[256 + 10];
    char va_buf[256];
    char *timestamp;
    va_list ap;

    if (!sp->buffer)
	return;

    va_start(ap, str);
    vsprintf (va_buf, str, ap);
    va_end(ap);
    timestamp = summary_plot_get_timestamp();
    snprintf(buf, sizeof(buf), "[%s] %s\n", timestamp, va_buf);

    gtk_text_buffer_get_end_iter(sp->buffer, &iter);
    gtk_text_buffer_insert(sp->buffer, &iter, buf, -1);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(sp->text),
				 gtk_text_buffer_get_mark(sp->buffer,
							  "insert"), 0.0,
				 FALSE, 0.0, 0.0);
    util_safe_free(timestamp);
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
    GtkBox *vbox;
    GtkNotebook *nb;
    GtkFrame *frame;
    GtkWidget *text;
    GdkColor color;
    GtkTextBuffer *buffer;
    GtkScrolledWindow *sw;

    summary_plot_type *sp;
    summary_plot_type *sp_gurbat;
    plot_dataset_type *d;
    int i, N;
    char dirstr[PATH_MAX];
    char filestr[PATH_MAX];
    double *x, *y;
    double x_max, y_max;


    plparseopts(&argc, (const char **) argv, PL_PARSE_FULL);
    gtk_init(&argc, &argv);

    /* 
     * PUNQS3
     */
    sp = summary_plot_alloc();
    sp->item = plot_alloc();
    plot_initialize(sp->item, NULL, NULL, CANVAS);
    plot_summary_collect_data(&x, &y, &N,
			      "/h/masar/EnKF_PUNQS3/PUNQS3/Original/PUNQS3.DATA",
			      "FOPT");
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, LINE);
    plot_dataset_add(sp->item, d);
    plot_util_get_maxima(sp->item, &x_max, &y_max);
    plot_set_viewport(sp->item, 0, x_max, 0, y_max);
    plot_data(sp->item);
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_001",
				     "PUNQS3_0001.DATA", "FOPT");
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_002",
				     "PUNQS3_0002.DATA", "FOPT");
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_095",
				     "PUNQS3_0095.DATA", "FOPT");
    summary_plot_add_ensamble_member(sp,
				     "/d/felles/bg/scratch/masar/member_090",
				     "PUNQS3_0090.DATA", "FOPT");

    /* 
     * GURBAT 
     */
    sp_gurbat = summary_plot_alloc();
    sp_gurbat->item = plot_alloc();
    plot_initialize(sp_gurbat->item, NULL, NULL, CANVAS);
    plot_get_canvas(sp_gurbat->item);
    plot_summary_collect_data(&x, &y, &N,
			      "/h/masar/EnKF_Martin/model/EXAMPLE_01_BASE.DATA",
			      "WOPR:OP_1");
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, LINE);
    plot_dataset_add(sp_gurbat->item, d);
    plot_util_get_maxima(sp_gurbat->item, &x_max, &y_max);
    plot_set_viewport(sp_gurbat->item, 0, x_max, 0, y_max);
    plot_data(sp_gurbat->item);
    for (i = 1; i < 5; i++) {
	snprintf(dirstr, PATH_MAX,
		 "/h/masar/EnKF_Martin/enkf_runs/member00%d", i);
	snprintf(filestr, PATH_MAX, "MARTIN_01_BASE-000%d.DATA", i);
	summary_plot_add_ensamble_member(sp_gurbat, dirstr, filestr,
					 "WOPR:OP_1");
    }

    /* 
     * START PACKING THE GUI 
     */
    win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_resize(GTK_WINDOW(win), 1024, 1000);
    gtk_container_set_border_width(GTK_CONTAINER(win), 0);
    g_signal_connect(G_OBJECT(win), "destroy", G_CALLBACK(destroy_local),
		     NULL);
    vbox = GTK_BOX(gtk_vbox_new(FALSE, 10));

    nb = GTK_NOTEBOOK(gtk_notebook_new());
    gtk_notebook_append_page(nb, GTK_WIDGET(plot_get_canvas(sp->item)),
			     NULL);
    gtk_notebook_append_page(nb,
			     GTK_WIDGET(plot_get_canvas(sp_gurbat->item)),
			     NULL);

    gtk_box_pack_start(vbox, GTK_WIDGET(nb), FALSE, FALSE, 0);

    frame = GTK_FRAME(gtk_frame_new(NULL));
    sw = GTK_SCROLLED_WINDOW(gtk_scrolled_window_new(NULL, NULL));
    text = gtk_text_view_new();
    gtk_container_add(GTK_CONTAINER(frame), GTK_WIDGET(sw));
    gtk_container_add(GTK_CONTAINER(sw), text);
    gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_IN);
    gtk_container_set_border_width(GTK_CONTAINER(frame), 5);
    gtk_container_set_border_width(GTK_CONTAINER(text), 5);
    gtk_text_view_set_editable(GTK_TEXT_VIEW(text), FALSE);
    gtk_text_view_set_justification(GTK_TEXT_VIEW(text), GTK_JUSTIFY_LEFT);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(text), GTK_WRAP_WORD);

    gdk_color_parse("gray", &color);
    gtk_widget_modify_base(text, GTK_STATE_NORMAL, &color);
    gtk_widget_modify_base(text, GTK_STATE_ACTIVE, &color);
    gtk_widget_modify_base(text, GTK_STATE_PRELIGHT, &color);
    gtk_widget_modify_base(text, GTK_STATE_SELECTED, &color);
    gtk_widget_modify_base(text, GTK_STATE_INSENSITIVE, &color);

    buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(text));

    gtk_box_pack_start(vbox, GTK_WIDGET(frame), TRUE, TRUE, 0);
    gtk_container_add(GTK_CONTAINER(win), GTK_WIDGET(vbox));

    sp->buffer = buffer;
    sp_gurbat->buffer = buffer;
    sp->text = text;
    sp_gurbat->text = text;

    summary_plot_append_textbox(sp, "Initialized gtk and plplot: %s...", "lalalal");

    summary_plot_timout(sp);
    g_timeout_add(5000, summary_plot_timout, sp);
    summary_plot_timout(sp_gurbat);
    g_timeout_add(5000, summary_plot_timout, sp_gurbat);
    gtk_widget_show_all(win);
    gtk_main();

    summary_plot_free(sp);
    summary_plot_free(sp_gurbat);

    return true;
}
