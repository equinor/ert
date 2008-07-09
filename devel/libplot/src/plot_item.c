#include "plot.h"

static plot_item *plot_item_alloc() {
     plot_item *item;

     item = malloc(sizeof *item);
     
     if (!item)
       return NULL;

     return item;
}

static plot_dataset *plot_dataset_alloc() {
     plot_dataset *dataset;

     dataset = malloc(sizeof *dataset);

     if (!dataset)
       return NULL;

     return dataset;
}

/**************************************************************/
/**************************************************************/

void plot_item_free(list_type *list, plot_item *item) { 
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



plot_item *plot_item_new(plot *p, char *dev, char *filename) {
     plot_item *item;

     item = plot_item_alloc();
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

     /* The following code switches BLACK and WHITE's int values.
      * By doing this we get a white background, and we can use plcol0()
      * to set BLACK values for axis as such.
      */
     plscol0(WHITE, 255, 255, 255);
     plscol0(BLACK, 0, 0, 0); 

     plinit();

     return item;
}

int plot_item_plot_data(plot *p, plot_item *item) {
     list_node_type *node , *next_node;

     node = list_get_head(item->datasets);
     while (node != NULL) {
	  plot_dataset *tmp;

	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);

	  printf("%s: adding a graph with %d samples in the x-y vectors\n", __func__, tmp->length);
	  
	  /* Set the defined color for the graph 
	   * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/color.html#color-map-0
	   */
	  plcol0((PLINT) tmp->color);
	  
	  switch (tmp->style) {
	     case HISTOGRAM:
		tmp->style = HISTOGRAM;
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

     return true;
}

void plot_item_set_graph_data(plot *p, plot_item *item, double *xvalue, double *yvalue, int length, plot_color color, plot_style style) {
     plot_dataset *dataset;
     dataset = plot_dataset_alloc();
     dataset->xvalue = xvalue;
     dataset->yvalue = yvalue;
     dataset->length = length;
     dataset->color = color;
     dataset->style = style;
     
     printf("%s: Adding dataset %p to list with length %d\n", __func__, dataset, dataset->length);
     list_append_ref(item->datasets, dataset);
}

void plot_item_set_labels(plot_item *item, char *xlabel, char *ylabel, char *title, plot_color color) {
     item->xlabel = strdup(xlabel);
     item->ylabel = strdup(ylabel);
     item->title = strdup(title);
     item->label_color = color;
     printf("%s: setting the labels for the plot\n", __func__);
}

void plot_item_set_viewport(plot_item *item, PLFLT xmin, PLFLT xmax, PLFLT ymin, PLFLT ymax) {
     printf("%s: setting the viewport for the plot\n", __func__);

     /* Advance to the next subpage, looks like it has to be done */
     pladv(0);

     /* Setup the viewport 
      * Device-independent routine for setting up the viewport
      * plvpor (xmin, xmax, ymin, ymax);
      * or just setup/define the standard viewport ....
      */
     plvsta();

     /*
      * Specify world coordinates of viewport boundaries
      * plwind (xmin, xmax, ymin, ymax);
      */
     plwind(xmin, xmax, ymin, ymax);

     /* Define a default color for the axis 
      * For some strange reason this won't work with BLACK.
      */
     plcol0(BLACK); 

      /* Draw a box with axes, etc.
       * plbox (xopt, xtick, nxsub, yopt, ytick, nysub); 
       * options at:
       * http://plplot.sourceforge.net/docbook-manual/plplot-html-5.9.0/plbox.html
       */
     //plbox("bnst", 0, 0, "bnstv", 0, 0);
     plschr(0, 0.6); 
     plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);

     if (!item->xlabel || !item->ylabel || !item->title) {
	  fprintf(stderr, "Error: you need to set lables before setting the viewport!\n");
     } else {
	  /* 
	   * http://old.ysn.ru/docs/plplot/plplotdoc-html-0.4.1/characters.html
	   */
	  plcol0(item->label_color);

	  /* Scale the textsize by 0.8 */
	  plschr(0, 0.8); 
	  /* Set some default values for the labels */
	  pllab(item->xlabel, item->ylabel, item->title);
     }

}

void plot_item_manipulate_data(plot *p, plot_item *item, void (*func)(void *data)) {
     item->func = func;
     item->func(item);
}

