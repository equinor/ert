#include <plot.h>
#include <plot_dataset.h>

struct plot_dataset_struct {
    double *xvalue;
    double *yvalue;
    double std_y;
    int length;
    plot_style_type style;
    plot_color_type color;
};

int plot_datset_get_length(plot_dataset_type * d)
{
    if (!d)
	return -1;

    return d->length;
}

plot_color_type plot_datset_get_color(plot_dataset_type * d)
{
    if (!d)
	return -1;

    return d->color;
}

plot_style_type plot_datset_get_style(plot_dataset_type * d)
{
    if (!d)
	return -1;

    return d->style;
}

double *plot_datset_get_vector_x(plot_dataset_type * d)
{
    if (!d)
	return NULL;

    return d->xvalue;
}

double *plot_datset_get_vector_y(plot_dataset_type * d)
{
    if (!d)
	return NULL;

    return d->yvalue;
}

void plot_datset_set_style(plot_dataset_type * d, plot_style_type s)
{
    if (!d)
	return;

    d->style = s;
}

plot_dataset_type *plot_dataset_alloc()
{
    plot_dataset_type *d;

    d = malloc(sizeof *d);
    if (!d)
	return NULL;

    return d;
}

void plot_dataset_free(plot_dataset_type * d)
{
    util_safe_free(d->xvalue);
    util_safe_free(d->yvalue);
    util_safe_free(d);
}

void
plot_dataset_set_data(plot_dataset_type * d, double *x, double *y,
		      int len, plot_color_type c, plot_style_type s)
{
    if (!d) {
	fprintf(stderr,
		"Error: you need to allocate the new dataset first\n");
	return;
    }

    d->xvalue = x;
    d->yvalue = y;
    d->length = len;
    d->color = c;
    d->style = s;
}

int plot_dataset_add(plot_type * item, plot_dataset_type * d)
{
    if (!d || !item) {
	fprintf(stderr,
		"Error: you need to allocate a new dataset or plot-item.\n");
	return false;
    }

    if (!d->xvalue || !d->yvalue || !d->length) {
	fprintf(stderr, "Error: you need to set the data first\n");
	return false;
    }

    printf("ID[%d] %s: Adding dataset %p to list with length %d\n",
	   plot_get_stream(item), __func__, d, d->length);
    list_append_ref(plot_get_datasets(item), d);

    return true;
}
