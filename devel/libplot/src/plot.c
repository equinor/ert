#include <plot.h>
#include <plot_dataset.h>

plot_type *plot_alloc()
{
    plot_type *item;

    item = malloc(sizeof *item);
    if (!item)
	return NULL;

    return item;
}

void plot_initialize(plot_type * item, char *dev, char *filename)
{
    item->device = strdup(dev);
    item->filename = strdup(filename);

    /*
     * Initialize these to avoid uninitialised conditional jumps 
     */
    item->xlabel = NULL;
    item->ylabel = NULL;
    item->title = NULL;

    item->datasets = list_alloc();

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
    fprintf(stderr, "%s: free on %p, (%s)\n", __func__, item,
	    item->filename);

    /*
     * Free the graphs in the plot 
     */
    plot_free_all_datasets(item);

    /*
     * Now free the rest 
     */
    util_safe_free(item->filename);
    util_safe_free(item->device);
    util_safe_free(item->xlabel);
    util_safe_free(item->ylabel);
    util_safe_free(item->title);

    util_safe_free(item);

    /*
     * Call the plplot end routine for this item 
     */
    plend();
}

void plot_data(plot_type * item)
{
    list_node_type *node, *next_node;

    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);

	printf("%s: adding a graph with %d samples in the x-y vectors\n",
	       __func__, tmp->length);

	/*
	 * Set the defined color for the graph
	 * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/color.html#color-map-0 
	 */
	plcol0((PLINT) tmp->color);

	switch (tmp->style) {
	case HISTOGRAM:
	    tmp->style = HISTOGRAM;
	    break;
	case LINE:
	    tmp->style = LINE;
	    plline(tmp->length, tmp->xvalue, tmp->yvalue);
	    break;
	case POINT:
	    tmp->style = POINT;
	    plpoin(tmp->length, tmp->xvalue, tmp->yvalue, 1);
	    break;
	default:
	    fprintf(stderr, "Error: no plot style is defined!\n");
	    break;
	}

	node = next_node;
    }

    printf("%s: finished plotting %s\n", __func__, item->filename);
}

void
plot_set_labels(plot_type * item, char *xlabel, char *ylabel,
		char *title, plot_color_type color)
{
    item->xlabel = strdup(xlabel);
    item->ylabel = strdup(ylabel);
    item->title = strdup(title);
    item->label_color = color;
    printf("%s: setting the labels for the plot\n", __func__);
}

void
plot_set_viewport(plot_type * item, PLFLT xmin, PLFLT xmax,
		  PLFLT ymin, PLFLT ymax)
{
    printf("%s: setting the viewport for the plot\n", __func__);

    /*
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
		"Error: you need to set lables before setting the viewport!\n");
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
