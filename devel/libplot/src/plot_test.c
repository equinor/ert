#include "plot.h"

void __plot_example_plot2(void *data) {
     plot_item *item = data;
     list_node_type *node , *next_node;
     int i;

     printf("%s: Manipulating data for example plot 2 (%s)\n", __func__, item->filename);

     node = list_get_head(item->datasets);
     while (node != NULL) {
	  plot_dataset *tmp;
	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);
	 
	  /* Scale up the y-axis by 50% */ 
	  for(i = 0; i < tmp->length; i++) {
	       tmp->yvalue[i] = tmp->yvalue[i] * 1.5;
	  }

	  node = next_node;
     }
}


int main(int argc, const char **argv) {
     plot_type *p;
     plot_item *item;
     int N = 100; /* Number of samples */
     const double period = 2*PI;
     int i;
     double *x, *y;
     
     /* Initialize the "engine" */
     p = plot_alloc();
     plot_init(p, true, argc, argv);

     printf ("------------------------------------------------\n");

     /* Create a new plot window, and fill with 2 graphs */
     item = plot_item_new(p, "png", "martin.png");
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = sin(x[i]);
     }
     plot_item_set_graph_data(p, item, x, y, N);

     /* Create another graph in the same plot */
     N = 50;
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = cos(x[i]);
     }
     plot_item_set_graph_data(p, item, x, y, N);
     plot_item_set_size(item, 0, period, -1, 1);
     plot_item_set_labels(item, "x-axis", "y-axis", "f(x) = sin(x) and f(x) = cos(x)");
     plot_item_plot_data(p, item);

     printf ("------------------------------------------------\n");
     
     /* Create a second new plot window, and fill it with only 1 graph  */
     item = plot_item_new(p, "jpeg", "plot.jpg");
     x = malloc(sizeof(double) * N);
     y = malloc(sizeof(double) * N);
     for (i = 0; i < N; i++) {
	  x[i] = i * period / N;
	  y[i] = exp(x[i]);
     }
     plot_item_set_graph_data(p, item, x, y, N);
     plot_item_manipulate_data(p, item, __plot_example_plot2);
     plot_item_set_size(item, 0, period, 0, 500 * 0.5);
     plot_item_set_labels(item, "x-axis", "y-axis", "f(x) = exp(x)");
     plot_item_plot_data(p, item);

     printf ("------------------------------------------------\n");

     plot_free(p);

     return 0;
}

