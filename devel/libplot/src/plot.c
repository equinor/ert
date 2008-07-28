#include <plot.h>
#include <plot_dataset.h>
#include <pthread.h>

#define WIDTH 1024
#define HEIGHT 740


/**
 * @brief Contains information about a plotting window.
 */
struct plot_struct {
    /**
     * This is a doubly linked list which contains the datasets
     * that will be plotted in this plot-window.
     */
    list_type *datasets;
    const char *filename; /**< Filename for the plot */

    /** Device name for the plot, where you have the following 
     * list of choices:
     * - xwin:       X-Window (Xlib)
     * - tk:         Tcl/TK Window
     * - gcw:        Gnome Canvas Widget
     * - ps:         PostScript File (monochrome)
     * - psc:        PostScript File (color)
     * - xfig:       Fig file
     * - hp7470:     HP 7470 Plotter File (HPGL Cartridge, Small Plotter)
     * - hp7580:     HP 7580 Plotter File (Large Plotter)
     * - lj_hpgl:    HP Laserjet III, HPGL emulation mode
     * - pbm:        PDB (PPM) Driver
     * - png:        PNG file
     * - jpeg:       JPEG file
     * - null:       Null device
     * - tkwin:      New tk driver
     * - mem:        User-supplied memory device
     * - gif:        GIF file
     * - svg:        Scalable Vector Graphics (SVG 1.1)
     *
     * @remarks: this is set with plot_initialize().
     */
    const char *device;
    int stream;	 /**< The plots current stream ID */
    int *tot_streams;  /**< Keeps strack of total streams */

    const char *xlabel;	 /**< Label for the x-axis */
    const char *ylabel;	 /**< Label for the y-axis */
    const char *title;	/**< Plot title */
    plot_color_type label_color;  /**< Color for the labels */
    plot_window_type w;
    PlplotCanvas *canvas;
};


/**
 * @return Returns a new plot_type pointer.
 * @brief Create a new plot_type
 *
 * Create a new plot - allocates the memory.
 */
plot_type *plot_alloc()
{
    plot_type *item;

    item = malloc(sizeof *item);
    if (!item)
	return NULL;

    item->w = NORMAL;
    item->stream = -1;

    return item;
}

int plot_get_stream(plot_type * item)
{
    if (!item)
	return -1;

    return item->stream;
}

void plot_set_stream(plot_type * item, int stream) 
{
    item->stream = stream;
}

plot_window_type plot_get_window_type(plot_type * item)
{
    if (!item)
	return -1;

    return item->w;

}

PlplotCanvas *plot_get_canvas(plot_type * item)
{
    if (!item)
	return NULL;

    return item->canvas;
}

list_type *plot_get_datasets(plot_type * item)
{
    if (!item)
	return NULL;

    return item->datasets;
}

/**
 * @brief Initialize a new plot
 * @param item your current plot
 * @param dev the output device
 * @param filename the output filename 
 * 
 * This function has to be called before you set any other information 
 * on the plot_type *item. This is because plinit() gets called inside this function.
 */
void plot_initialize(plot_type * item, const char *dev,
		     const char *filename, plot_window_type w)
{
    static int output_stream = 0;
    static int tot_streams = 0;
    pthread_mutex_t update_lock;

    item->stream = output_stream;
    item->tot_streams = &tot_streams;
    fprintf(stderr, "ID[%d] SETTING UP NEW OUTPUT STREAM\n", item->stream);

    if (item->w != CANVAS)
        plsstrm(item->stream);

    item->xlabel = NULL;
    item->ylabel = NULL;
    item->title = NULL;
    item->w = w;
    item->datasets = list_alloc();

    /* Define unique output stream */

    item->device = dev;
    item->filename = filename;

    if (dev)
	plsdev(item->device);
    if (dev) {
	if (strcmp(item->device, "xwin"))
	    plsfnam(item->filename);
    }

    /*
     * The following code switches BLACK and WHITE's int values. By doing
     * this we get a white background, and we can use plcol0() to set
     * BLACK values for axis as such. 
     */
    if (item->w == CANVAS) {
	item->canvas = plplot_canvas_new();
	plplot_canvas_set_size(item->canvas, WIDTH, HEIGHT);
	plplot_canvas_use_persistence(item->canvas, TRUE);
	plplot_canvas_scol0(item->canvas, WHITE, 255, 255, 255);
	plplot_canvas_scol0(item->canvas, BLACK, 0, 0, 0);
    } else {
	plscol0(WHITE, 255, 255, 255);
	plscol0(BLACK, 0, 0, 0);
	plinit();
    }

    /* Make this threadsafe */
    pthread_mutex_init(&update_lock, NULL);
    pthread_mutex_lock(&update_lock);
    /* Use another input stream next time */
    output_stream++;
    tot_streams++;
    pthread_mutex_unlock(&update_lock);
}

/**
 * @brief Free all the datasets in a plot item
 * @param item your current plot
 * 
 * Using this function is optional, the datasets WILL be freed  when plot_free() is called anyway.
 */
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

/**
 * @brief Free your plot item
 * @param item your current plot
 * 
 * Use this function to free your allocated memory from plot_alloc().
 */
void plot_free(plot_type * item)
{
    pthread_mutex_t update_lock;

    fprintf(stderr, "ID[%d] %s: free on %p, (%s)\n", item->stream,
	    __func__, item, item->filename);

    /*
     * Free the graphs in the plot 
     */
    plot_free_all_datasets(item);

    /* First we end the current stream, then we check if the program is on the last
     * active stream. If so, cleanup everything..
     */
    if (item->w == CANVAS) {
        printf("Stream number: %d\n", plplot_canvas_get_stream_number(item->canvas));
/*
        plsstrm(plplot_canvas_get_stream_number(item->canvas));
        plend1();
*/
        plplot_canvas_finalize(item->canvas);
    } else {
        plsstrm(item->stream);
        plend1();
    }

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

/**
 * @brief Do the actual plotting
 * @param item your current plot
 * 
 * After adding the datasets, setting viewport and labels you are ready to do the plotting.
 */
void plot_data(plot_type * item)
{
    list_node_type *node, *next_node;

    plsstrm(item->stream);
    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);

	printf
	    ("ID[%d] %s: plotting graph with %d samples in the x-y vectors\n",
	     item->stream, __func__, plot_datset_get_length(tmp));

	plot_dataset(item, tmp);
	node = next_node;
    }

}

/**
 * @brief Set labels
 * @param item your current plot
 * @param xlabel label for the x-axis
 * @param ylabel label for the y-axis
 * @param title title for the plot
 * @param color the color for your labels
 * 
 * Set the labels for your plot, do this before setting the viewport.
 */
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
plot_resize_axis(plot_type * item, PLFLT xmin, PLFLT xmax, PLFLT ymin,
		 PLFLT ymax)
{
    plsstrm(item->stream);

    plplot_canvas_col0(item->canvas, BLACK);
    plplot_canvas_adv(item->canvas, 0);
    plplot_canvas_vsta(item->canvas);
    plplot_canvas_wind(item->canvas, xmin, xmax, ymin, ymax);
    plplot_canvas_schr(item->canvas, 0, 0.5);
    plplot_canvas_box(item->canvas, "bcnst", 0.0, 0, "bcnstv", 0.0, 0);

    printf("Set canvas xmax %f, ymax: %f\n", xmax, ymax);

    plplot_canvas_adv(item->canvas, 0);

}

/**
 * @brief Setup viewport
 * @param item your current plot
 * @param xmin minimum value for the x-axis
 * @param xmax maximum value for the x-axis
 * @param ymin minimum value for the y-axis
 * @param ymax maximum value for the y-axis
 * 
 * Sets up your viewport, defining and the axis, setting up fonts and colors.
 */
void
plot_set_viewport(plot_type * item, PLFLT xmin, PLFLT xmax,
		  PLFLT ymin, PLFLT ymax)
{
    printf("ID[%d] %s: setting the viewport for the plot\n", item->stream,
	   __func__);
    plsstrm(item->stream);

    if (item->w == CANVAS) {
	plot_resize_axis(item, xmin, xmax, ymin, ymax);

	plplot_canvas_col0(item->canvas, item->label_color);
        plplot_canvas_schr(item->canvas, 0, 0.5);
        plplot_canvas_lab(item->canvas, item->xlabel, item->ylabel, item->title);

	plplot_canvas_adv(item->canvas, 0);
    } else {

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

	    plschr(0, 0.5);
	    /*
	     * Set some default values for the labels 
	     */
	    pllab(item->xlabel, item->ylabel, item->title);
	}
    }
}


void plot_get_maxima(plot_type * item, double *x_max, double *y_max)
{
    list_node_type *node, *next_node;
    double tmp_x = 0;
    double tmp_y = 0;
    double *x, *y;
    int i;

    node = list_get_head(plot_get_datasets(item));
    while (node != NULL) {
	plot_dataset_type *tmp;
	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
        x = plot_datset_get_vector_x(tmp);
        y = plot_datset_get_vector_y(tmp);

        for (i = 0; i <= plot_datset_get_length(tmp); i++) {
            if (x[i] > tmp_x)
                tmp_x = x[i];
            if (y[i] > tmp_y)
                tmp_y = y[i];
        }
	node = next_node;
    }
    
    fprintf(stderr, "ID[%d] Found maxima: x: %f and y: %f\n", plot_get_stream(item), tmp_x, tmp_y);
    *x_max = tmp_x;
    *y_max = tmp_y;
}
