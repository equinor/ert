#include <plot.h>
#include <plot_dataset.h>
#include <plot_canvas.h>
#include <glib.h>
#include <gtk/gtk.h>
#include <plplot/plplotcanvas.h>


void destroy_local(GtkWidget * widget, gpointer data)
{
    gtk_main_quit();
}

int main(int argc, char *argv[])
{
    GtkWidget *win;
    PlplotCanvas *canvas;
    PlplotCanvas *canvas2;
    GtkBox *vbox;
    GtkBox *hbox;
    GtkButton *button;
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
    gtk_box_pack_start(vbox, GTK_WIDGET(canvas), TRUE, FALSE, 0);
    gtk_box_pack_start(vbox, GTK_WIDGET(canvas2), TRUE, FALSE, 0);
    hbox = GTK_BOX(gtk_hbox_new(FALSE, 0));
    button = GTK_BUTTON(gtk_button_new_from_stock(GTK_STOCK_EXECUTE));
    gtk_box_pack_start(hbox, GTK_WIDGET(button), TRUE, FALSE, 0);
    gtk_box_pack_start(vbox, GTK_WIDGET(hbox), TRUE, FALSE, 0);
    win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_container_set_border_width(GTK_CONTAINER(win), 5);
    g_signal_connect(G_OBJECT(win), "destroy", G_CALLBACK(destroy_local),
		     item);
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
    plot_dataset_set_data(d, x, y, (N*2), GREEN, LINE);
    plot_dataset_add(item2, d);

    /* 
     * PLOT THE DATA WITH TIMER FUNCTIONS
     */
    g_timeout_add(100, plot_canvas_data_join, item);
    g_timeout_add(100, plot_canvas_data_join, item2);

    gtk_widget_show_all(win);
    gtk_main();

    plot_free(item);
    plot_free(item2);

    return 0;
}
