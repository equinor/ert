#include <plot.h>
#include <plot_dataset.h>
#include <assert.h>


/**
 * @brief Contains information about a dataset.
 */
struct plot_dataset_struct {
    PLFLT *xvalue; /**< Vector containing x-axis data */
    PLFLT *yvalue; /**< Vector containing y-axis data */
    int length;	/**< Length of the vectors defining the axis */
    plot_style_type style; /**< The graph style */
    plot_color_type color; /**< The graph color */
    int step; /**< Helper value when using animation */
    bool finished; /**< Helper value when using animation */
};


void plot_dataset_finished(plot_dataset_type * d, bool flag)
{
    assert(d != NULL);
    d->finished = flag;
}


bool plot_dataset_is_finished(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->finished;
}

int plot_dataset_get_step(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->step;
}

int plot_dataset_step_next(plot_dataset_type * d)
{
    assert(d != NULL);
    d->step++;
    return d->step;
}

int plot_datset_get_length(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->length;
}

plot_color_type plot_datset_get_color(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->color;
}

plot_style_type plot_datset_get_style(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->style;
}

PLFLT *plot_datset_get_vector_x(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->xvalue;
}

PLFLT *plot_datset_get_vector_y(plot_dataset_type * d)
{
    assert(d != NULL);
    return d->yvalue;
}

/**
 * @return Returns a new plot_dataset_type pointer.
 * @brief Create a new plot_dataset_type
 *
 * Create a new dataset - allocates the memory.
 */
plot_dataset_type *plot_dataset_alloc()
{
    plot_dataset_type *d;

    d = malloc(sizeof *d);
    return d;
}

/**
 * @brief Free your dataset item
 * @param d your current dataset
 *
 * Use this function to free your allocated memory from plot_dataset_alloc().
 */
void plot_dataset_free(plot_dataset_type * d)
{
    assert(d != NULL);
    util_safe_free(d->xvalue);
    util_safe_free(d->yvalue);
    util_safe_free(d);
}

/**
 * @brief Set the collected data to the dataset.
 * @param d your current dataset
 * @param x vector containing x-data
 * @param y vector containing y-data
 * @param len length of vectors
 * @param c color for the graph
 * @param s style for the graph
 *
 * After collecting your x-y data you have to let the dataset item know about
 * it. At the same time you define some detail about how the graph should look.
 */
void
plot_dataset_set_data(plot_dataset_type * d, PLFLT * x, PLFLT * y,
		      int len, plot_color_type c, plot_style_type s)
{
    assert(d != NULL);
    len = len + 1;
    d->xvalue = malloc(sizeof(PLFLT) * len);
    memcpy(d->xvalue, x, sizeof(PLFLT) * len);
    if (y) {
	d->yvalue = malloc(sizeof(PLFLT) * len);
	memcpy(d->yvalue, y, sizeof(PLFLT) * len);
    } else {
	d->yvalue = NULL;
    }
    d->length = len - 1;
    d->color = c;
    d->style = s;
    d->step = 0;
    d->finished = false;
}


/*
  This function seems to work only canvas ...
void plot_dataset_join(plot_type * item, plot_dataset_type * d, int from,
		       int to)
{
    int i, k, k2;
    PLFLT *x = d->xvalue;
    PLFLT *y = d->yvalue;

    assert(item != NULL && d != NULL);
    plsstrm(plot_get_stream(item));
    printf("item: %p, dataset: %p, FROM %d\t TO: %d\n", item, d, from, to);

    for (i = 0; i < (to - from); i++) {
	k = from + i;
	k2 = k + 1;
	printf("plotting from %d -> %d: %f, %f to %f, %f\n",
	       k, k2, x[k], y[k], x[k2], y[k2]);
	plplot_canvas_join(plot_get_canvas(item),
			   x[k], y[k], x[k2], y[k2]);
	plplot_canvas_adv(plot_get_canvas(item), 0);
    }

}
*/


void plot_dataset(plot_type * item, plot_dataset_type * d)
{
    assert(item != NULL && d != NULL);
    plsstrm(plot_get_stream(item));
    plcol0((PLINT) plot_datset_get_color(d));


    switch (plot_datset_get_style(d)) {
    case LINE:
      plline(plot_datset_get_length(d),
	     plot_datset_get_vector_x(d),
	     plot_datset_get_vector_y(d));
	break;
    case POINT:
      plssym(0, SYMBOL_SIZE);
      plpoin(plot_datset_get_length(d),
	     plot_datset_get_vector_x(d),
	     plot_datset_get_vector_y(d), SYMBOL);
      break;
    case BLANK:
	break;
    default:
	break;
    }
}


/**
 * @brief Add a dataset to the plot 
 * @param item your current plot
 * @param d your current dataset
 *
 * When the data is in place in the dataset you can add it to the plot item,
 * this way it will be included when you do the plot.
 */
int plot_dataset_add(plot_type * item, plot_dataset_type * d)
{
    int i;

    if (!d || !item) {
	fprintf(stderr,
		"Error: you need to allocate a new dataset or plot-item.\n");
	return false;
    }

    if (!d->xvalue || !d->yvalue || !d->length) {
	fprintf(stderr, "Error: you need to set the data first\n");
	return false;
    }

    i = list_get_size(plot_get_datasets(item));
    list_append_ref(plot_get_datasets(item), d);
    assert(i <= list_get_size(plot_get_datasets(item)));

    return true;
}

/**
 * @brief Get extrema values from one dataset
 * @param d your current dataset
 * @param x_max pointer to max x-value
 * @param y_max pointer to max y-value
 * @param x_min pointer to the new x minimum
 * @param y_min pointer to the new y minimum
 * 
 * Find the extrema values in the plot item, checks all added dataset.
 */
void plot_dataset_get_extrema(plot_dataset_type * d, double *x_max,
			      double *y_max, double *x_min, double *y_min)
{
    double tmp_x_max = 0;
    double tmp_y_max = 0;
    double tmp_x_min = 0;
    double tmp_y_min = 0;
    int i;
    double *x, *y;
    bool flag = false;

    assert(d != NULL);
    x = plot_datset_get_vector_x(d);
    y = plot_datset_get_vector_y(d);

    for (i = 0; i <= plot_datset_get_length(d); i++) {
	if (!flag) {
	    tmp_x_max = x[i];
	    tmp_x_min = x[i];
	    tmp_y_max = y[i];
	    tmp_y_min = y[i];
	    flag = true;
	}
	if (y[i] > tmp_y_max)
	    tmp_y_max = y[i];
	if (y[i] < tmp_y_min)
	    tmp_y_min = y[i];
	if (x[i] > tmp_x_max)
	    tmp_x_max = x[i];
	if (x[i] < tmp_x_min)
	    tmp_x_min = x[i];
    }
    if (x_max)
	*x_max = tmp_x_max;
    if (y_max)
	*y_max = tmp_y_max;
    if (x_min)
	*x_min = tmp_x_min;
    if (y_min)
	*y_min = tmp_y_min;
}

plot_dataset_type *plot_dataset_get_prominent(plot_type * item, int *len)
{
    list_node_type *node, *next_node;
    plot_dataset_type *ref = NULL;
    int tmp_len = 0;

    assert(item != NULL);

    node = list_get_head(plot_get_datasets(item));
    while (node != NULL) {
	plot_dataset_type *tmp;
	int len2;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
	len2 = plot_datset_get_length(tmp);
	if (len2 > tmp_len) {
	    ref = tmp;
	    tmp_len = len2;
	}
	node = next_node;
    }
    if (len)
	*len = tmp_len;

    return ref;
}
