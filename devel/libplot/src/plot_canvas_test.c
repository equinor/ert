#include <plot.h>
#include <plot_dataset.h>
#include <glib.h>
#include <gtk/gtk.h>
#include <plplot/plplotcanvas.h>

PlplotCanvas *canvas;
PlplotCanvas *canvas2;

gboolean plot_canvas_data_join(gpointer data)
{
    plot_type *item = data;
    list_node_type *node, *next_node;
    int len;
    double *x;
    double *y;
    int step;
    int flag = true;

//    plplot_canvas_clear(plot_get_canvas(item));
    plplot_canvas_adv(plot_get_canvas(item), 0);

    node = list_get_head(plot_get_datasets(item));
    while (node != NULL) {
        plot_dataset_type *tmp;
        next_node = list_node_get_next(node);
        tmp = list_node_value_ptr(node);
        len = plot_datset_get_length(tmp);

        if (plot_dataset_is_finished(tmp)) {
            node = next_node;
            continue;
        }
        flag = false;

        if (plot_dataset_get_step(tmp) == len) {
            printf("ID[%d] Plotted last node, skipping..\n",
                   plot_get_stream(item));
            plot_dataset_finished(tmp, true);
            node = next_node;
            continue;
        }

        step = plot_dataset_step_next(tmp);

        x = plot_datset_get_vector_x(tmp);
        y = plot_datset_get_vector_y(tmp);
        plot_dataset_join(item, tmp, step - 1, step);

        printf("ID[%d] Plotting step %d -> %d of total %d\n",
               plot_get_stream(item), step - 1, step, len);
        node = next_node;
    }

    if (!flag)
        return true;

    return false;
}

void destroy_local(GtkWidget * widget, gpointer data)
{
    gtk_main_quit();
}

void resize_window(GtkWidget * widget, gpointer data)
{
    GtkRequisition requisition;
    const double period = 2 * PI;
    int w = ((GtkWidget *)data)->requisition.width;
    int h = ((GtkWidget *)data)->requisition.height;

    printf("foo %dx%d\n", ((GtkWidget *)data)->allocation.height, ((GtkWidget *)data)->allocation.width);

//  plplot_canvas_set_size(canvas2, h, w);
/*
    plplot_canvas_vsta(canvas2);
    plplot_canvas_wind(canvas2, -period, period*2, -0.3, 1);
    plplot_canvas_clear(canvas2);
    plplot_canvas_box(canvas2, "bcnst", 0.0, 0, "bcnstv", 0.0, 0);
    plplot_canvas_adv(canvas2, 0);
*/
}

int main(int argc, char *argv[])
{
    GtkWidget *win;
    GtkBox *vbox;
    GtkBox *hbox;
    plot_type *item;
    plot_type *item2;
    plot_dataset_type *d;
    int N = 30;
    const double period = 2 * PI;
    int i;
    double *x, *y;

    plparseopts(&argc, (const char **) argv, PL_PARSE_FULL);
    gtk_init(&argc, &argv);

    /* 
     * CREATE TWO PLOT ITEMS
     */
    item = plot_alloc();
    plot_initialize(item, NULL, NULL, CANVAS);
    canvas = plot_get_canvas(item);
    plot_set_viewport(item, 0, 2 * PI, -1, 1);

    item2 = plot_alloc();
    plot_initialize(item2, NULL, NULL, CANVAS);
    canvas2 = plot_get_canvas(item2);
    plot_set_viewport(item2, -period, period, -0.3, 1);

    /* 
     * START GTK PACKING CODE 
     */
    vbox = GTK_BOX(gtk_vbox_new(FALSE, 0));
    //gtk_box_pack_start(vbox, GTK_WIDGET(canvas), TRUE, TRUE, 0);
    gtk_box_pack_start(vbox, GTK_WIDGET(canvas2), TRUE, TRUE, 0);
    hbox = GTK_BOX(gtk_hbox_new(FALSE, 0));
    gtk_box_pack_start(vbox, GTK_WIDGET(hbox), TRUE, TRUE, 0);
    win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_container_set_border_width(GTK_CONTAINER(win), 5);
    g_signal_connect(G_OBJECT(win), "destroy", G_CALLBACK(destroy_local),
		     item);
/*
    g_signal_connect(G_OBJECT(win), "check_resize", G_CALLBACK(resize_window),
		     vbox);
*/
    gtk_container_add(GTK_CONTAINER(win), GTK_WIDGET(vbox));

    /* 
     * START CREATING PLOT DATA
     */
    x = malloc(sizeof(double) * (N + 1));
    y = malloc(sizeof(double) * (N + 1));
    for (i = 0; i <= N; i++) {
        x[i] = i * period / N;
        y[i] = sin(x[i]);
    }
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLUE, LINE);
    plot_dataset_add(item, d);

    x = malloc(sizeof(double) * ((N*2) + 1));
    y = malloc(sizeof(double) * ((N*2) + 1));
    for (i = 0; i <= N*2; i++) {
        x[i] = i * period / (N*2);
        y[i] = cos(x[i]);
    }
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, (N*2), RED, LINE);
    plot_dataset_add(item, d);

    x = malloc(sizeof(double) * ((N*2) + 1));
    y = malloc(sizeof(double) * ((N*2) + 1));
    for (i = 0; i <= N*2; i++) {
        x[i] = (i - N) / period;
        if (x[i] != 0.0)
            y[i] = sin(PI * x[i]) / (PI * x[i]);
        else
            y[i] = 1.0;
    }
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, (N*2), BLACK, LINE);
    plot_dataset_add(item2, d);

    /* 
     * PLOT THE DATA WITH TIMER FUNCTIONS
     */
//    g_timeout_add(100, plot_canvas_data_join, item);
    plplot_canvas_use_persistence(plot_get_canvas(item2), FALSE);
    g_timeout_add(100, plot_canvas_data_join, item2);

    gtk_widget_show_all(win);
    gtk_main();

    plot_free(item);
    plot_free(item2);

    return 0;
}
