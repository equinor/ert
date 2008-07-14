#include <plot.h>
#include <plot_dataset.h>
#include <pthread.h>


struct plot_struct {
    list_type *datasets;
    const char *filename;
    const char *device;
    int stream;
    int *tot_streams;

    const char *xlabel;
    const char *ylabel;
    const char *title;
    plot_color_type label_color;
};


plot_type *plot_alloc()
{
    plot_type *item;

    item = malloc(sizeof *item);
    if (!item)
	return NULL;

    return item;
}

int plot_get_stream(plot_type * item)
{
    if (!item)
	return -1;

    return item->stream;
}

list_type *plot_get_datasets(plot_type * item)
{
    if (!item)
	return NULL;

    return item->datasets;
}

void plot_initialize(plot_type * item, const char *dev,
		     const char *filename)
{
    static int output_stream = 0;
    static int tot_streams = 0;
    pthread_mutex_t update_lock;

    item->stream = output_stream;
    item->tot_streams = &tot_streams;
    fprintf(stderr, "ID[%d] SETTING UP NEW OUTPUT STREAM\n", item->stream);
    plsstrm(item->stream);

    item->xlabel = NULL;
    item->ylabel = NULL;
    item->title = NULL;
    item->datasets = list_alloc();

    /* Define unique output stream */

    item->device = dev;
    item->filename = filename;
    plsdev(item->device);
    plsfnam(item->filename);

    /*
     * The following code switches BLACK and WHITE's int values. By doing
     * this we get a white background, and we can use plcol0() to set
     * BLACK values for axis as such. 
     */
    plscol0(WHITE, 255, 255, 255);
    plscol0(BLACK, 0, 0, 0);
    plinit();

    /* Make this threadsafe */
    pthread_mutex_init(&update_lock, NULL);
    pthread_mutex_lock(&update_lock);
    /* Use another input stream next time */
    output_stream++;
    tot_streams++;
    pthread_mutex_unlock(&update_lock);
}

void plot_free_all_datasets(plot_type * item)
{
    list_node_type *node, *next_node;

    if (!item->datasets)
	return;

    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
	list_del_node(item->datasets, node);
	plot_dataset_free(tmp);
	node = next_node;
    }

    list_free(item->datasets);
    item->datasets = NULL;

}

void plot_free(plot_type * item)
{
    pthread_mutex_t update_lock;

    fprintf(stderr, "ID[%d] %s: free on %p, (%s)\n", item->stream,
	    __func__, item, item->filename);

    plsstrm(item->stream);

    /*
     * Free the graphs in the plot 
     */
    plot_free_all_datasets(item);

    /* First we end the current stream, then we check if the program is on the last
     * active stream. If so, cleanup everything..
     */
    plend1();
    fprintf(stderr, "ID[%d] ENDING CURRENT STREAM\n", item->stream);


    pthread_mutex_init(&update_lock, NULL);
    pthread_mutex_lock(&update_lock);
    /* We now have one less stream */
    --*item->tot_streams;
    if (!*item->tot_streams) {
	fprintf(stderr, "ID[%d] Cleaning up!\n", item->stream);
	plend();
    }
    pthread_mutex_unlock(&update_lock);

    util_safe_free(item);
}

void plot_data(plot_type * item)
{
    list_node_type *node, *next_node;
    printf("ID[%d] %s: plotting %s\n", item->stream, __func__,
	   item->filename);


    plsstrm(item->stream);
    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);

	printf
	    ("ID[%d] %s: plotting graph with %d samples in the x-y vectors\n",
	     item->stream, __func__, plot_datset_get_length(tmp));

	/*
	 * Set the defined color for the graph
	 * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/color.html#color-map-0 
	 */
	plcol0((PLINT) plot_datset_get_color(tmp));

	switch (plot_datset_get_style(tmp)) {
	case HISTOGRAM:
	    plot_datset_set_style(tmp, HISTOGRAM);
	    break;
	case LINE:
	    plot_datset_set_style(tmp, LINE);
	    plline(plot_datset_get_length(tmp),
		   plot_datset_get_vector_x(tmp),
		   plot_datset_get_vector_y(tmp));
	    break;
	case POINT:
	    plot_datset_set_style(tmp, POINT);
	    plpoin(plot_datset_get_length(tmp),
		   plot_datset_get_vector_x(tmp),
		   plot_datset_get_vector_y(tmp), 1);
	    break;
	default:
	    fprintf(stderr, "Error: no plot style is defined!\n");
	    break;
	}

	node = next_node;
    }

}

void
plot_set_labels(plot_type * item, const char *xlabel, const char *ylabel,
		const char *title, plot_color_type color)
{
    printf("ID[%d] %s: setting the labels for the plot\n", item->stream,
	   __func__);

    item->xlabel = xlabel;
    item->ylabel = ylabel;
    item->title = title;
    item->label_color = color;
}

void
plot_set_viewport(plot_type * item, PLFLT xmin, PLFLT xmax,
		  PLFLT ymin, PLFLT ymax)
{
    printf("ID[%d] %s: setting the viewport for the plot\n", item->stream,
	   __func__);
    plsstrm(item->stream);

    /* DOCUMENTATION:
     * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/viewport_window.html#viewports
     *
     * Advance to the next subpage, looks like it has to be done 
     */
    pladv(0);

    /*
     * Setup the viewport Device-independent routine for setting up the
     * viewport plvpor (xmin, xmax, ymin, ymax); or just setup/define the
     * standard viewport .... 
     */
    plvsta();

    /*
     * Specify world coordinates of viewport boundaries
     * plwind (xmin, xmax, ymin, ymax);
     */
    plwind(xmin, xmax, ymin, ymax);

    /*
     * Define a default color for the axis For some strange reason this
     * won't work with BLACK. 
     */
    plcol0(BLACK);

    /*
     * Draw a box with axes, etc. plbox (xopt, xtick, nxsub, yopt, ytick,
     * nysub); options at:
     * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/plbox.html 
     */
    plschr(0, 0.6);
    plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);

    if (!item->xlabel || !item->ylabel || !item->title) {
	fprintf(stderr,
		"!!!! ID[%d] ERROR: you need to set lables before setting the viewport!\n",
		item->stream);
    } else {
	/*
	 * http://old.ysn.ru/docs/plplot/plplotdoc-html-0.4.1/characters.html
	 */
	plcol0(item->label_color);

	/*
	 * Scale the textsize by 0.8 
	 */
	plschr(0, 0.8);
	plfont(1);
	/*
	 * Set some default values for the labels 
	 */
	pllab(item->xlabel, item->ylabel, item->title);
    }

}
