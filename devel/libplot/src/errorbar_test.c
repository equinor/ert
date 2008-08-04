#include <plot.h>
#include <plot_dataset.h>

#define LEN 10
#define DATASETS 8

void plot1()
{
    plot_type *item;
    plot_dataset_type *d;

    srand(time(NULL));

    item = plot_alloc();
    plot_initialize(item, "png", "stat.png", NORMAL);
    {
	int i, j;
	PLFLT *y, *x;

	for (j = 0; j <= DATASETS; j++) {
	    y = malloc(sizeof(PLFLT));
	    x = malloc(sizeof(PLFLT));
	    for (i = 0; i <= LEN; i++) {
		if (i > 0) {
		    y = realloc(y, sizeof(PLFLT) * (i + 1));
		    x = realloc(x, sizeof(PLFLT) * (i + 1));
		}
		x[i] = i;
		y[i] = rand() % 100;
	    }
	    d = plot_dataset_alloc();
	    plot_dataset_set_data(d, x, y, LEN, BLACK, POINT);
	    plot_dataset_add(item, d);
	    util_safe_free(y);
	    util_safe_free(x);
	}
    }

    plot_set_labels(item, "x-axis", "y-axis", "Test", BLACK);
    plot_set_viewport(item, 0 - 1, LEN, 0 - 10, 100 + 10);
    plot_data(item);
    plot_errorbar_data(item);
    plot_free(item);
}

PLFLT calc_rms(PLFLT * y, int len, PLFLT mean)
{
    int i;
    PLFLT diff;
    PLFLT sum = 0;

    for (i = 0; i < len; i++) {
	diff = y[i] - mean;
	sum += pow(diff, 2);
    }
    return sqrtf((1 / (PLFLT) len) * sum);
}


void plot2()
{
    plot_type *item;
    plot_dataset_type *d;
    int N = pow(2, 4);
    const PLFLT period = 2 * M_PI;
    PLFLT x[N];
    PLFLT y[N];
    int i;
    PLFLT rms, mean;
    PLFLT sum = 0;
    double xmax, ymax, xmin, ymin;

    srand48(time(NULL));

    item = plot_alloc();
    plot_initialize(item, "png", "std.png", NORMAL);

    for (i = 0; i < N; i++) {
	x[i] = (i * period) / N;
	y[i] = drand48();
	sum += y[i];
    }
    mean = sum / N;
    rms = calc_rms(y, N, mean);

    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLACK, LINE);
    plot_dataset_add(item, d);

    for (i = 0; i < N; i++)
	y[i] += rms;
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLUE, BLANK);
    plot_dataset_add(item, d);

    for (i = 0; i < N; i++)
	y[i] -= 2 * rms;
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, N, BLUE, BLANK);
    plot_dataset_add(item, d);

    plot_set_labels(item, "x-axis", "y-axis",
		    "Standard deviation - drand48()", BLACK);
    plot_get_extrema(item, &xmax, &ymax, &xmin, &ymin);
    plot_set_viewport(item, xmin, xmax, ymin, ymax);
    plot_data(item);
    plot_errorbar_data(item);
    plot_free(item);
}

int main(int argc, const char **argv)
{
    plot1();
    plot2();

    return 0;
    argc = 0;
    argv = NULL;
}
