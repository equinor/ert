#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>
#include <plot_summary.h>
#include <glib.h>
#include <gtk/gtk.h>
#include <ecl_kw.h>
#include <ecl_sum.h>
#include <list.h>
#include <list_node.h>
#include <config.h>
#include <time.h>

#define TIMEOUT 10000
#define ECL_EXT ".DATA"

/***************************************************************
 ***************************************************************/

typedef enum summary_plot_enkf_enum {
    POST = 0,
    PRIOR = 1,
    REAL = 2,
} summary_plot_enkf_type;

typedef struct summary_plot_gui_struct {
    GtkWidget *win;
    GtkTextBuffer *buffer;
    GtkWidget *text;
    GtkNotebook *nb;
    config_type *config;
    const char *conf_file;
    list_type *list;
} summary_plot_gui_type;

typedef struct summary_plot_struct {
    summary_plot_gui_type *spg;	/* "parent" pointer */
    plot_type *item;
    list_node_type *node;
    list_type *list;
    double x_max;
    double y_max;
    int timer_id;
    char *kw;
    bool off_charts;
    bool delete;
} summary_plot_type;


typedef struct summary_plot_member_struct {
    int files;
    char *dir;
    char *file;
    int last_report_step;
    summary_plot_enkf_type t;
} summary_plot_member_type;

/***************************************************************
 ***************************************************************/

static summary_plot_type *summary_plot_alloc();
static void summary_plot_free(summary_plot_type * sp);
static summary_plot_member_type *summary_plot_member_alloc();
static void summary_plot_member_free(summary_plot_member_type * spm);
static summary_plot_gui_type *summary_plot_gui_alloc();
static void summary_plot_gui_free(summary_plot_gui_type * spg);
static void summary_plot_get_ecl_data(summary_plot_member_type * spm,
				      char ***sfl, char **header_file,
				      int *files);
static void summary_plot_add_ensamble_member(summary_plot_type * sp,
					     char *dir, char *file,
					     summary_plot_enkf_type t);
static gboolean summary_plot_timout(gpointer data);
static char *summary_plot_get_timestamp();
static void summary_plot_append_textbox(summary_plot_type * sp,
					const char *str, ...);
static config_type *summary_plot_init_config(const char *config_file);
static void summary_plot_initialize_ensembles(summary_plot_type * sp,
					      char *sp_kw);
static void summary_plot_setup_gui(summary_plot_gui_type * spg);
static summary_plot_type
    * summary_plot_create_tab_with_data(summary_plot_gui_type * spg,
					char *sp_kw, double xmax,
					double ymax);
void summary_plot_add_well_tabs(summary_plot_gui_type * spg,
				char *conf_file);
static void summary_plot_destroy_local(GtkWidget * widget, gpointer data);
static void summary_plot_exit(summary_plot_gui_type * spg);

/***************************************************************
 ***************************************************************/

summary_plot_type *summary_plot_alloc()
{
    summary_plot_type *sp;
    sp = malloc(sizeof *sp);
    sp->list = list_alloc();
    sp->off_charts = false;
    sp->delete = false;
    sp->timer_id = -1;
    sp->node = NULL;
    return sp;
}

void summary_plot_free(summary_plot_type * sp)
{
    list_node_type *node, *next_node;

    if (list_get_size(sp->list) != 0) {
	node = list_get_head(sp->list);
	while (node != NULL) {
	    summary_plot_member_type *tmp;
	    next_node = list_node_get_next(node);
	    tmp = list_node_value_ptr(node);
	    list_del_node(sp->list, node);
	    summary_plot_member_free(tmp);

	    node = next_node;
	}
    }

    list_free(sp->list);
    plot_free(sp->item);
    util_safe_free(sp->kw);
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
    util_safe_free(spm);
}

summary_plot_gui_type *summary_plot_gui_alloc()
{
    summary_plot_gui_type *spg;
    spg = malloc(sizeof *spg);
    spg->list = list_alloc();
    return spg;
}

void summary_plot_gui_free(summary_plot_gui_type * spg)
{
    config_free(spg->config);
    util_safe_free(spg);
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
				      char *dir, char *file,
				      summary_plot_enkf_type t)
{
    summary_plot_member_type *spm;

    spm = summary_plot_member_alloc();
    spm->dir = strdup(dir);
    spm->file = strdup(file);
    spm->t = t;
    summary_plot_append_textbox(sp,
				"Adding ensemble %s/%s to plot %p with keyword '%s'",
				dir, file, sp->item, sp->kw);

    summary_plot_get_ecl_data(spm, NULL, NULL, &spm->files);
    list_append_ref(sp->list, spm);
}


gboolean summary_plot_timout(gpointer data)
{
    list_node_type *node, *next_node;
    list_node_type *node2, *next_node2;
    summary_plot_gui_type *spg = data;
    summary_plot_type *sp_new;
    char **summary_file_list;
    char *header_file;
    int j;
    ecl_sum_type *ecl_sum;
    int report_step, first_report_step, last_report_step;
    PLFLT *x, *y;
    PLFLT diff_day;
    time_t t, t0;
    plot_dataset_type *d;
    double x_max, y_max;

    node2 = list_get_head(spg->list);
    while (node2 != NULL) {
	bool flag = false;
	summary_plot_type *sp;
	next_node2 = list_node_get_next(node2);
	sp = list_node_value_ptr(node2);

	printf("--\n");
	printf("Handling %s, stream: %d\n", sp->kw,
	       plplot_canvas_get_stream_number(plot_get_canvas(sp->item)));

	node = list_get_head(sp->list);
	while (node != NULL) {
	    summary_plot_member_type *tmp;
	    next_node = list_node_get_next(node);
	    tmp = list_node_value_ptr(node);

	    summary_plot_get_ecl_data(tmp, &summary_file_list,
				      &header_file, &j);

	    if (tmp->files != j || tmp->last_report_step == 0) {

		ecl_sum =
		    ecl_sum_fread_alloc(header_file, j, (const char **)
					summary_file_list, true, true);
		ecl_sum_get_report_size(ecl_sum, &first_report_step,
					&last_report_step);

		x = malloc(sizeof(PLFLT) * (last_report_step + 1));
		y = malloc(sizeof(PLFLT) * (last_report_step + 1));

		for (report_step = first_report_step;
		     report_step <= last_report_step; report_step++) {
		    if (ecl_sum_has_report_nr(ecl_sum, report_step)) {
			int day, month, year;
			util_set_date_values(ecl_sum_get_sim_time
					     (ecl_sum, report_step), &day,
					     &month, &year);

			if (report_step == first_report_step)
			    plot_util_get_time(day, month, year, &t0,
					       NULL);
			if (!t0) {
			    fprintf(stderr,
				    "!!!! Error: no first report step was found\n");
			    continue;
			}

			plot_util_get_time(day, month, year, &t, NULL);
			plot_util_get_diff(&diff_day, t, t0);

			x[report_step] = (PLFLT) diff_day;
			y[report_step] = (PLFLT)
			    ecl_sum_get_general_var(ecl_sum, report_step,
						    sp->kw);
		    }
		}


		/* If this is the first time - we want to plot up to the current step */
		if (tmp->last_report_step == 0) {
		    d = plot_dataset_alloc();
		    if (tmp->t == POST)
			plot_dataset_set_data(d, x, y, last_report_step,
					      RED, LINE);
		    else if (tmp->t == PRIOR)
			plot_dataset_set_data(d, x, y, last_report_step,
					      BLUE, LINE);
		    else
			plot_dataset_set_data(d, x, y, last_report_step,
					      BLACK, LINE);

		    plot_dataset_get_maxima(d, &x_max, &y_max);
		    if (x_max > sp->x_max) {
			sp->x_max = x_max;
			flag = true;
		    }
		    if (y_max > sp->y_max) {
			sp->y_max = y_max;
			flag = true;
		    }

		    plot_dataset(sp->item, d);
		    plot_dataset_free(d);
		    summary_plot_append_textbox(sp,
						"Plotting dataset in plot %p (%s), until report step %d.",
						sp->item, tmp->dir,
						last_report_step - 1);
		} else {
		    /* Join lines between all the next points (steps) */
		    d = plot_dataset_alloc();
		    printf("Stream: %d dataset: %p KW: %s dir: %s\n",
			   plot_get_stream(sp->item), d, sp->kw, tmp->dir);
		    plot_dataset_set_data(d, x, y, last_report_step, RED,
					  LINE);
		    printf("ymax: %f, xmax: %f\n", sp->y_max, sp->x_max);
		    plot_dataset_join(sp->item, d,
				      tmp->last_report_step - 1,
				      last_report_step - 1);
		    plot_dataset_get_maxima(d, &x_max, &y_max);
		    if (x_max > sp->x_max) {
			sp->x_max = x_max;
			flag = true;
		    }
		    if (y_max > sp->y_max) {
			sp->y_max = y_max;
			flag = true;
		    }
		    summary_plot_append_textbox(sp,
						"Plotting dataset segment in plot %p (%s), from report step %d to %d",
						sp->item, tmp->dir,
						tmp->last_report_step - 1,
						last_report_step - 1);
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

	if (flag) {
	    int i;

	    printf
		("We are now drawing off the axis, creating new plot, deleting old!\n");

	    sp_new =
		summary_plot_create_tab_with_data(sp->spg, sp->kw,
						  sp->x_max, sp->y_max);
	    printf
		("!!! Adding a new tab %p with kw: %s and new xmax: %f, ymax: %f\n",
		 sp_new->item, sp->kw, sp->x_max, sp->y_max);

	    printf("Prev stream: %d\n", plot_get_stream(sp->item));
	    printf("New stream: %d\n", plot_get_stream(sp_new->item));

	    i = gtk_notebook_page_num(GTK_NOTEBOOK(sp->spg->nb),
				      GTK_WIDGET(plot_get_canvas
						 (sp->item)));
	    /* MEMLEAK! plplot_canvas_finalize gets called here, but it looks
	       like there is a memleak here somewhere in plplot :-) */

	    list_del_node(spg->list, sp->node);
	    summary_plot_free(sp);
	    gtk_notebook_remove_page(GTK_NOTEBOOK(sp_new->spg->nb), i);
	    gtk_widget_show_all(sp_new->spg->win);
	}

	node2 = next_node2;
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
    strftime(str, sizeof(str), "%T", ptr);

    return strdup(str);
}

void summary_plot_append_textbox(summary_plot_type * sp, const char *str,
				 ...)
{
    GtkTextIter iter;
    char buf[512 + 10];
    char va_buf[512];
    char *timestamp;
    va_list ap;

    if (!sp->spg->buffer)
	return;

    va_start(ap, str);
    vsprintf(va_buf, str, ap);
    va_end(ap);
    timestamp = summary_plot_get_timestamp();
    snprintf(buf, sizeof(buf), "[%s] %s\n", timestamp, va_buf);

    gtk_text_buffer_get_end_iter(sp->spg->buffer, &iter);
    gtk_text_buffer_insert(sp->spg->buffer, &iter, buf, -1);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(sp->spg->text),
				 gtk_text_buffer_get_mark(sp->spg->buffer,
							  "insert"), 0.0,
				 FALSE, 0.0, 0.0);
    util_safe_free(timestamp);
}


config_type *summary_plot_init_config(const char *config_file)
{
    config_type *config;

    config = config_alloc(false);
    config_init_item(config, "DATA_FILE", 0, NULL, true, false, 0, NULL, 1,
		     1, NULL);
    config_init_item(config, "ECL_STORE_PATH", 0, NULL, true, false, 0,
		     NULL, 1, 1, NULL);
    config_init_item(config, "ECLBASE", 0, NULL, true, false, 0, NULL, 1,
		     1, NULL);
    config_init_item(config, "ECL_STORE", 0, NULL, true, false, 0, NULL, 1,
		     -1, NULL);
    config_init_item(config, "WELL", 0, NULL, true, true, 2, NULL, 1,
		     -1, NULL);

    config_parse(config, config_file, ECL_COM_KW);

    {
	/* Change Path to your enkf config dir */
	char *path;
	util_alloc_file_components(config_file, &path, NULL, NULL);
	chdir(path);
	util_safe_free(path);
    }

    return config;
}

void summary_plot_initialize_ensembles(summary_plot_type * sp, char *sp_kw)
{
    /* Collecting data about the ensembles and add them to the plot */
    config_item_type *config_item;
    const char **argv_list;
    const char *ecl_store_path;
    const char *ecl_base;
    char *ecl_store_path_buf;
    char *ecl_base_buf;
    char *base_with_ext;
    int n, i, j;

    ecl_store_path = config_get(sp->spg->config, "ECL_STORE_PATH");
    ecl_base = config_get(sp->spg->config, "ECLBASE");
    config_item = config_get_item(sp->spg->config, "ECL_STORE");
    argv_list = config_item_get_argv(config_item, &n);
    for (i = 1; i < n; i++) {
	if (*argv_list[i] == ',')
	    continue;
	if (*argv_list[i] == '-') {
	    /* This hack can't handle spaces in the "int-int , int-int" format! */
	    for (j = atoi(argv_list[i - 1]);
		 j <= atoi(argv_list[i + 1]); j++) {
		ecl_store_path_buf = malloc(strlen(ecl_store_path) + 1);
		snprintf(ecl_store_path_buf,
			 (int) strlen(ecl_store_path) + 1,
			 ecl_store_path, j);

		ecl_base_buf = malloc(strlen(ecl_base) + 1);
		snprintf(ecl_base_buf, (int) strlen(ecl_base) + 1,
			 ecl_base, j);

		base_with_ext =
		    malloc(strlen(ecl_base_buf) + strlen(ECL_EXT) + 1);
		snprintf(base_with_ext,
			 (int) strlen(ecl_base_buf) + strlen(ECL_EXT) +
			 2, "%s%s", ecl_base_buf, ECL_EXT);

		summary_plot_add_ensamble_member(sp,
						 ecl_store_path_buf,
						 base_with_ext, POST);
		util_safe_free(ecl_store_path_buf);
		util_safe_free(ecl_base_buf);
		util_safe_free(base_with_ext);
	    }
	}
    }

}

void summary_plot_setup_gui(summary_plot_gui_type * spg)
{
    GtkBox *vbox;
    GtkFrame *frame;
    GtkScrolledWindow *sw;

    spg->win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_resize(GTK_WINDOW(spg->win), 1024, 1000);
    gtk_container_set_border_width(GTK_CONTAINER(spg->win), 0);
    g_signal_connect(G_OBJECT(spg->win), "destroy",
		     G_CALLBACK(summary_plot_destroy_local), spg);
    vbox = GTK_BOX(gtk_vbox_new(FALSE, 10));

    spg->nb = GTK_NOTEBOOK(gtk_notebook_new());
    gtk_box_pack_start(vbox, GTK_WIDGET(spg->nb), FALSE, FALSE, 0);
    frame = GTK_FRAME(gtk_frame_new(NULL));
    sw = GTK_SCROLLED_WINDOW(gtk_scrolled_window_new(NULL, NULL));
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
				   GTK_POLICY_AUTOMATIC,
				   GTK_POLICY_AUTOMATIC);
    spg->text = gtk_text_view_new();
    gtk_container_add(GTK_CONTAINER(frame), GTK_WIDGET(sw));
    gtk_container_add(GTK_CONTAINER(sw), spg->text);
    gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_IN);
    gtk_container_set_border_width(GTK_CONTAINER(frame), 2);
    gtk_container_set_border_width(GTK_CONTAINER(spg->text), 2);
    gtk_text_view_set_editable(GTK_TEXT_VIEW(spg->text), FALSE);
    gtk_text_view_set_justification(GTK_TEXT_VIEW(spg->text),
				    GTK_JUSTIFY_LEFT);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(spg->text), GTK_WRAP_WORD);
    spg->buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(spg->text));
    gtk_box_pack_start(vbox, GTK_WIDGET(frame), TRUE, TRUE, 0);
    gtk_container_add(GTK_CONTAINER(spg->win), GTK_WIDGET(vbox));
}

summary_plot_type *summary_plot_create_tab_with_data(summary_plot_gui_type
						     * spg, char *sp_kw,
						     double xmax,
						     double ymax)
{
    /* Setup a plot object and plot the true case */
    summary_plot_type *sp;
    int N;
    PLFLT *x, *y;
    const char *ecl_data_file;
    plot_dataset_type *d;
    int stream;

    ecl_data_file = config_get(spg->config, "DATA_FILE");

    sp = summary_plot_alloc();
    sp->spg = spg;

    sp->kw = strdup(sp_kw);
    sp->item = plot_alloc();
    plot_initialize(sp->item, NULL, NULL, CANVAS);
    plot_summary_collect_data(&x, &y, &N, ecl_data_file, sp_kw);
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, POINT);
    plot_dataset_add(sp->item, d);

    if ((xmax == 0) || (ymax == 0))
	plot_get_maxima(sp->item, &sp->x_max, &sp->y_max);
    else {
	sp->x_max = xmax;
	sp->y_max = ymax;
    }

    if (sp->x_max == 0 || sp->y_max == 0) {
	fprintf(stderr, "Error: maxima for either x or y axis is zero!\n");
	exit(-1);
    }
    plot_set_labels(sp->item, "Days", sp_kw, spg->conf_file, BLACK);
    plot_set_viewport(sp->item, 0, sp->x_max, 0, sp->y_max);
    plot_data(sp->item);
    gtk_notebook_append_page(spg->nb,
			     GTK_WIDGET(plot_get_canvas
					(sp->item)), gtk_label_new(sp_kw));
    summary_plot_initialize_ensembles(sp, sp_kw);
    summary_plot_append_textbox(sp,
				"Adding timer for %p with timeout %d ms",
				sp->item, TIMEOUT);

    stream = plplot_canvas_get_stream_number(plot_get_canvas(sp->item));
    printf("Created plot item with stream id: %d\n", stream);

    sp->node = list_append_ref(spg->list, sp);

    return sp;
}

void summary_plot_add_well_tabs(summary_plot_gui_type * spg,
				char *conf_file)
{
    FILE *stream;
    bool at_eof = false;

    spg->conf_file = conf_file;

    stream = util_fopen(conf_file, "r");
    while (!at_eof) {
	int i, tokens;
	int active_tokens;
	char **token_list;
	char *line;

	line = util_fscanf_alloc_line(stream, &at_eof);
	if (line != NULL) {
	    util_split_string(line, " \t", &tokens, &token_list);

	    active_tokens = tokens;
	    for (i = 0; i < tokens; i++) {
		char *comment_ptr = NULL;
		comment_ptr = strstr(token_list[i], ECL_COM_KW);

		if (comment_ptr != NULL) {
		    if (comment_ptr == token_list[i])
			active_tokens = i;
		    else
			active_tokens = i + 1;
		    break;
		}
	    }

	    if (active_tokens > 0) {
		if (!strcmp("WELL", token_list[0])) {
		    printf("Found a WELL (%s) with elements:\n",
			   token_list[1]);
		    for (i = 2; i < tokens; i++) {
			char buf[128];
			snprintf(buf, sizeof(buf), "%s:%s",
				 token_list[i], token_list[1]);
			summary_plot_create_tab_with_data(spg, buf, 0, 0);
		    }
		}
	    }
	}
	util_free_stringlist(token_list, tokens);
	util_safe_free(line);
    }

}


void summary_plot_destroy_local(GtkWidget * widget, gpointer data)
{
    summary_plot_gui_type *spg = data;
    summary_plot_exit(spg);
    gtk_main_quit();
}

void summary_plot_exit(summary_plot_gui_type * spg)
{
    list_node_type *node, *next_node;

    node = list_get_head(spg->list);
    while (node != NULL) {
	summary_plot_type *sp;

	next_node = list_node_get_next(node);
	sp = list_node_value_ptr(node);
	list_del_node(spg->list, node);
	summary_plot_free(sp);

	node = next_node;
    }

    list_free(spg->list);
    summary_plot_gui_free(spg);
}

/***************************************************************
 ***************************************************************/

int main(int argc, char **argv)
{
    summary_plot_gui_type *spg;

    if (argc < 2) {
	fprintf(stderr, "** ERROR ** %s EnKF.conf \n", argv[0]);
	exit(-1);
    }

    gtk_init(&argc, &argv);
    spg = summary_plot_gui_alloc();
    summary_plot_setup_gui(spg);

    spg->config = summary_plot_init_config(argv[1]);
    summary_plot_add_well_tabs(spg, argv[1]);

    summary_plot_timout(spg);
    g_timeout_add(TIMEOUT, summary_plot_timout, spg);

    gtk_widget_show_all(spg->win);
    gtk_main();

    return true;
}
