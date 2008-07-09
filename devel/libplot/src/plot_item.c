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
     plinit();

     return item;
}

int plot_item_plot_data(plot_type *p, plot_item *item, plot_style style) {
     list_node_type *node , *next_node;

     node = list_get_head(item->datasets);
     while (node != NULL) {
	  plot_dataset *tmp;

	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);

	  printf("%s: adding a graph with %d samples in the x-y vectors\n", __func__, tmp->length);
	  switch (style) {
	     case HISTOGRAM:
		/* plhist (n, data, datmin, datmax, nbin, opt); 
		 * n: number of data pointis
		 * data:    Pointer to array with values of the n data points.
		 * datamin: Left-hand edge of lowest-valued bin. 
		 * datmax:  Right-hand edge of highest-valued bin.
		 * nbin:    Number of (equal-sized) bins into which to divide the interval xmin to xmax.
		 * opt:     Options
		 *
		 * This isn't really needed for our purpose, a frequency - value 
		 * plot of some OPR might not be that interesting.
		 */
		break;
	     case LINE:
		plline(tmp->length, tmp->xvalue, tmp->yvalue);
		break;
	     case POINT:
		plpoin(tmp->length, tmp->xvalue, tmp->yvalue, 4);
		break;
	     default:
		break;
	  }

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
}

void plot_item_set_viewport(plot_item *item, PLFLT xmin, PLFLT xmax, PLFLT ymin, PLFLT ymax) {
     printf("%s: setting the viewport for the plot\n", __func__);

     /* Advance to the next subpage, looks like it has to be done */
     pladv(0);

     /* Setup the viewport 
      * Device-independent routine for setting up the viewport
      * plvpor (xmin, xmax, ymin, ymax);
      * plvpor(0.15, 0.85, 0.1, 0.9);
      * or just setup/define the standard viewport ....
      */
     plvsta();

     /*
      * Specify world coordinates of viewport boundaries
      * plwind (xmin, xmax, ymin, ymax);
      */
     plwind(xmin, xmax, ymin, ymax);

     /* Sets the color for color map0 
      * Se color choices:
      * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/color.html#color-map-0
      */
     plcol0(2);

      /* Draw a box with axes, etc.
       * plbox (xopt, xtick, nxsub, yopt, ytick, nysub); 
       * options at:
       * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/plbox.html
       */
     plbox("bnst", 0, 0, "bnstv", 0, 0);

     if (!item->xlabel || !item->ylabel || !item->title) {
	  fprintf(stderr, "Error: you need to set lables before setting the viewport!\n");
     } else {
	  pllab(item->xlabel, item->ylabel, item->title);
     }

}

void plot_item_manipulate_data(plot_type *p, plot_item *item, void (*func)(void *data)) {
     item->func = func;
     item->func(item);
}

