#include "plot.h"
#include "plot_dataset.h"

int main(int argc, const char **argv) {
     plot *item;
     plot_dataset *d;
     int N = 100; /* Number of samples */
     const double period = 2*PI;
     int i;
     double *x, *y;
     
     /* Create a new plot window, and fill with 2 graphs */
     item = plot_alloc();
     plot_initialize(item, "png", "martin.png");

     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = sin(x[i]);
     }
     d = plot_dataset_alloc();
     plot_dataset_set_data(d, x, y, N, BROWN, LINE);
     plot_dataset_add(item, d);

     /* Create another graph in the same plot */
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = cos(x[i]);
     }
     d = plot_dataset_alloc();
     plot_dataset_set_data(d, x, y, N, BROWN, LINE);
     plot_dataset_add(item, d);
    
     /* Create yet another cos, but with another angular frequency (\omega = 3) */ 
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * (period) / N;
	  y[i] = cos(3*x[i]);
     }
     d = plot_dataset_alloc();
     plot_dataset_set_data(d, x, y, N, RED, POINT);
     plot_dataset_add(item, d);

     plot_set_labels(item, "x-axis", "y-axis", "Harmonic waves", BROWN);
     plot_set_viewport(item, 0, period, -1, 1);
     plot_data(item);
     plot_free(item);

     printf ("------------------------------------------------\n");

     /* Create a second new plot window, and fill it with only 1 graph  */
     item = plot_alloc();
     plot_initialize(item, "jpeg", "plot.jpg");
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = exp(x[i]);
     }

     d = plot_dataset_alloc();
     plot_dataset_set_data(d, x, y, N, BLUE, LINE);
     plot_dataset_add(item, d);

     plot_set_labels(item, "x-axis", "y-axis", "f(x) = exp(x)", BROWN);
     plot_set_viewport(item, 0, period, 0, 500 * 0.5);
     plot_data(item);
     plot_free(item);

     printf ("------------------------------------------------\n");

     return 0;
}

