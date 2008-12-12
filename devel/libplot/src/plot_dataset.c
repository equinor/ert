#include <util.h>
#include <plot.h>
#include <plot_dataset.h>
#include <plot_const.h>
#include <assert.h>





/**
 * @brief Contains information about a dataset.
 */
struct plot_dataset_struct {
  double  	 *x; 	   	   /**< Vector containing x-axis data */
  double  	 *y; 	   	   /**< Vector containing y-axis data */
  double  	 *y1;               
  double  	 *y2;
  double  	 *x1;
  double  	 *x2;
  int     	  data_mask;       /**< An integer value written as a sum of values from the enum plot_data_types - says which type of data (x/y/...) is supplied. */
  plot_data_type type;             
  
  bool    shared_data;     /**< Whether this instance owns x,y,x1,... or just holds a reference. */
  int alloc_size;          /**< The allocated size of x,y (std) - will be 0 if shared_data = true. */
  int size;	   	   /**< Length of the vectors defining the plot */
  plot_style_type style;   /**< The graph style */
  plot_color_type color;   /**< The graph color */
};


static void plot_dataset_realloc_data(plot_dataset_type * d, int new_alloc_size) {
  if (d->data_mask & plot_data_x) d->x = util_realloc(d->x , new_alloc_size * sizeof * d->x , __func__);
  if (d->data_mask & plot_data_y) d->y = util_realloc(d->y , new_alloc_size * sizeof * d->y , __func__);

  if (d->data_mask & plot_data_x1) d->x1 = util_realloc(d->x1 , new_alloc_size * sizeof * d->x1 , __func__);
  if (d->data_mask & plot_data_x2) d->x2 = util_realloc(d->x2 , new_alloc_size * sizeof * d->x2 , __func__);

  if (d->data_mask & plot_data_y1) d->y1 = util_realloc(d->y1 , new_alloc_size * sizeof * d->y1 , __func__);
  if (d->data_mask & plot_data_y2) d->y2 = util_realloc(d->y2 , new_alloc_size * sizeof * d->y2 , __func__);

  d->alloc_size = new_alloc_size;
}



static void __append_vector(double * target , const double * src , int target_offset , int size) {
  if (src == NULL)
    util_abort("%s: trying to extract data from NULL pointer\n",__func__);
  
  memcpy(&target[target_offset] , src , size * sizeof * src);
}


static void plot_dataset_append_vector__(plot_dataset_type * d , int size , const double * x , const double * y , const double * y1 , const double * y2 , const double *x1 , const double *x2) {
  if (d->shared_data) 
    util_abort("%s: dataset has shared data - can not append \n",__func__);
  
  if (d->alloc_size < (d->size + size))
    plot_dataset_realloc_data(d , 2*(d->size + size));

  if (d->data_mask & plot_data_x)  __append_vector(d->x  , x  , d->size , size);
  if (d->data_mask & plot_data_x1) __append_vector(d->x1 , x1 , d->size , size);
  if (d->data_mask & plot_data_x2) __append_vector(d->x2 , x2 , d->size , size);

  if (d->data_mask & plot_data_y)  __append_vector(d->y  , y  , d->size , size);
  if (d->data_mask & plot_data_y1) __append_vector(d->y1 , y1 , d->size , size);
  if (d->data_mask & plot_data_y2) __append_vector(d->y2 , y2 , d->size , size);
  
  d->size += size;
}


int plot_dataset_get_length(plot_dataset_type * d)
{
  assert(d != NULL);
  return d->size;
}

plot_color_type plot_dataset_get_color(plot_dataset_type * d)
{
  assert(d != NULL);
  return d->color;
}

plot_style_type plot_dataset_get_style(plot_dataset_type * d)
{
  assert(d != NULL);
  return d->style;
}


double * plot_dataset_get_vector_x(const plot_dataset_type * d)  { return d->x; }
double * plot_dataset_get_vector_y(const plot_dataset_type * d)  { return d->y; }
double * plot_dataset_get_vector_x1(const plot_dataset_type * d) { return d->x1; }
double * plot_dataset_get_vector_y1(const plot_dataset_type * d) { return d->y1; }
double * plot_dataset_get_vector_x2(const plot_dataset_type * d) { return d->x2; }
double * plot_dataset_get_vector_y2(const plot_dataset_type * d) { return d->y2; }


/**
 * @return Returns a new plot_dataset_type pointer.
 * @brief Create a new plot_dataset_type
 *
 * Create a new dataset - allocates the memory.
 */
plot_dataset_type *plot_dataset_alloc(int data_mask , bool shared_data)
{
    plot_dataset_type *d;
    
    d = util_malloc(sizeof *d , __func__);
    d->data_mask   = data_mask;
    d->x   	   = NULL;
    d->y   	   = NULL;
    d->x1 	   = NULL;
    d->x2 	   = NULL;
    d->y1 	   = NULL;
    d->y2 	   = NULL;
    d->size        = 0;
    d->alloc_size  = 0;
    d->shared_data = shared_data;
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
  if (!d->shared_data) {
    util_safe_free(d->x);
    util_safe_free(d->y);
  }
  free(d);
}


void plot_dataset_set_shared_data(plot_dataset_type * d , int size , double *x , double *y) {
  if (d->shared_data) {
    d->x    = x;
    d->y    = y;
    d->size = size;
  } else
    util_abort("%s ... \n");
}


void plot_dataset_append_point(plot_dataset_type * d, double x , double y) {
  plot_dataset_append_vector__(d , 1 , &x , &y , NULL , NULL , NULL , NULL);
}


void plot_dataset_append_vector(plot_dataset_type * d, int size , const double * x , const double *y) {
  plot_dataset_append_vector__(d , size , x , y , NULL , NULL , NULL , NULL);
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
void plot_dataset_set_data(plot_dataset_type * d, const double * x, const double * y, int len, plot_color_type c, plot_style_type s)
     
{
  assert(d != NULL);
  plot_dataset_append_vector__(d , len , x , y , NULL , NULL , NULL , NULL);
  
  d->size = len;
  d->color = c;
  d->style = s;
}


/*
  This function seems to work only canvas ...
void plot_dataset_join(plot_type * item, plot_dataset_type * d, int from,
		       int to)
{
    int i, k, k2;
    double *x = d->x;
    double *y = d->y;

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
    plcol0((PLINT) plot_dataset_get_color(d));


    switch (plot_dataset_get_style(d)) {
    case LINE:
      plline(plot_dataset_get_length(d),
	     plot_dataset_get_vector_x(d),
	     plot_dataset_get_vector_y(d));
	break;
    case POINT:
      plssym(0, SYMBOL_SIZE);
      plpoin(plot_dataset_get_length(d),
	     plot_dataset_get_vector_x(d),
	     plot_dataset_get_vector_y(d), SYMBOL);
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

    if (!d->x || !d->y || !d->size) {
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
void plot_dataset_update_range(plot_dataset_type * d, bool first_pass , double *x_min, double *x_max,double *y_min, double *y_max)
{
  double tmp_x_max = *x_max;
  double tmp_y_max = *y_max;
  double tmp_x_min = *x_min;
  double tmp_y_min = *y_min;
  int i;
  double *x, *y;
  
  x   = plot_dataset_get_vector_x(d);
  y   = plot_dataset_get_vector_y(d);

  if (first_pass) {
    tmp_x_max = x[0];
    tmp_x_min = x[0];

    tmp_y_min = y[0];
    tmp_y_max = y[0];
  }
  
  for (i = 0; i < plot_dataset_get_length(d); i++) {
    if (y[i] > tmp_y_max)
      tmp_y_max = y[i];
    if (y[i] < tmp_y_min)
      tmp_y_min = y[i];
    
    
    if (x[i] > tmp_x_max)
      tmp_x_max = x[i];
    if (x[i] < tmp_x_min)
      tmp_x_min = x[i];
  }
  
  *x_max = tmp_x_max;
  *y_max = tmp_y_max;
  *x_min = tmp_x_min;
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
	len2 = plot_dataset_get_length(tmp);
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
