#include "plot.h"

plot_item *__plot_item_alloc() {
     plot_item *item;

     item = malloc(sizeof *item);
     
     if (!item)
       return NULL;

     return item;
}

plot_dataset *__plot_dataset_alloc() {
     plot_dataset *dataset;

     dataset = malloc(sizeof *dataset);

     if (!dataset)
       return NULL;

     return dataset;
}

void __plot_item_free(list_type *list, plot_item *item) { 
     list_node_type *node , *next_node;
     
     fprintf(stderr, "%s: free on %p, (%s)\n", __func__, item, item->filename);
    
     /* Free the graphs in the plot */ 
     node = list_get_head(item->datasets);
     while (node != NULL) {
	  plot_dataset *tmp;
	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);
	  list_del_node(item->datasets, node);
	  util_safe_free(tmp->xvalue);
	  util_safe_free(tmp->yvalue);
	  util_safe_free(tmp);

	  node = next_node;
     }
     list_free(item->datasets);
     
     /* Now free the rest */
     util_safe_free(item->filename);
     util_safe_free(item->device);
     util_safe_free(item->xlabel);
     util_safe_free(item->ylabel);
     util_safe_free(item->title);

     /* Delete the item-node from the main plot "windows" list */ 
     list_del_node(list, item->node);
     
     util_safe_free(item);

     /* Call the plplot end routine for this item */
     plend();
}



plot_item *plot_item_new(plot_type *p, char *dev, char *filename) {
     plot_item *item;

     item = __plot_item_alloc();
     item->device = strdup(dev);
     item->filename = strdup(filename);

     /* Initialize these to avoid uninitialised conditional jumps */
     item->xlabel = NULL;
     item->ylabel = NULL;
     item->title = NULL;

     item->datasets = list_alloc();

     item->node = list_append_ref(p->plots, item);
     plsfnam(item->filename);
     plsdev(item->device);
     plscolbg(255, 255, 255); /* RGB, white */
     plinit();
     plcol0(8);
     plschr(0.0, 0.8);
     plfont(1);

	  
     return item;
}

int plot_item_plot_data(plot_type *p, plot_item *item) {
     list_node_type *node , *next_node;

     node = list_get_head(item->datasets);
     while (node != NULL) {
	  plot_dataset *tmp;

	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);

	  printf("%s: adding a graph with %d samples in the x-y vectors\n", __func__, tmp->length);
	  plline(tmp->length, tmp->xvalue, tmp->yvalue);

	  node = next_node;
     }

     printf("%s: finished plotting %s\n", __func__, item->filename);

     __plot_item_free(p->plots, item);

     return true;
}

void plot_item_set_graph_data(plot_type *p, plot_item *item, double *xvalue, double *yvalue, int length) {
     plot_dataset *dataset;
     dataset = __plot_dataset_alloc();
     dataset->xvalue = xvalue;
     dataset->yvalue = yvalue;
     dataset->length = length;
     
     printf("%s: Adding dataset %p to list with length %d\n", __func__, dataset, dataset->length);
     list_append_ref(item->datasets, dataset);
}

void plot_item_set_labels(plot_item *item, char *xlabel, char *ylabel, char *title) {
     item->xlabel = strdup(xlabel);
     item->ylabel = strdup(ylabel);
     item->title = strdup(title);
     printf("%s: setting the labels for the plot\n", __func__);
     pllab(item->xlabel, item->ylabel, item->title);
}

void plot_item_set_size(plot_item *item, PLFLT xmin, PLFLT xmax, PLFLT ymin, PLFLT ymax) {
     printf("%s: setting the size for the plot\n", __func__);
     plenv(xmin, xmax, ymin, ymax, 0, 0);
}

void plot_item_manipulate_data(plot_type *p, plot_item *item, void (*func)(void *data)) {
     item->func = func;
     item->func(item);
}

