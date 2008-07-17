#include <plot.h>
#include <plot_dataset.h>

int main(int argc, const char **argv)
{
    plot_type *item;
    plot_dataset_type *d;
    const double period = 2 * PI;

    item = plot_alloc();
    plot_initialize(item, "xwin", NULL, NORMAL);

    {
	double *x, *y;
	int N = 100;
	int i;
	x = malloc(sizeof(double) * 2 * N);
	y = malloc(sizeof(double) * 2 * N);
	for (i = 0; i < 2 * N; i++) {
	    x[i] = (i - N) / period;
	    if (x[i] != 0.0)
		y[i] = sin(PI * x[i]) / (PI * x[i]);
	    else
		y[i] = 1.0;
	}
	d = plot_dataset_alloc();
	plot_dataset_set_data(d, x, y, 2 * N, BLUE, LINE);
	plot_dataset_add(item, d);
    }

    plot_set_labels(item, "x-axis", "y-axis", "#fry = sinc(x)", BLACK);
    plot_set_viewport(item, -period, period, -0.3, 1);
    plot_data(item);
    plot_free(item);

    return 0;
}
