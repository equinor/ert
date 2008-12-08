#include <assert.h>
#include <pthread.h>
#include <plot.h>
#include <plot_dataset.h>
#include <plot_util.h>
#include <plot_range.h>


/**
 * @brief Contains information about a plotting window.
 */
struct plot_struct {
    /**
     * This is a doubly linked list which contains the datasets
     * that will be plotted in this plot-window.
     */
    list_type *datasets;
    const char *filename; /**< Filename for the plot */

  /** Device name for the plot, where you have the following 
   * list of choices:
   * - xwin:       X-Window (Xlib)
   * - tk:         Tcl/TK Window
   * - gcw:        Gnome Canvas Widget
   * - ps:         PostScript File (monochrome)
   * - psc:        PostScript File (color)
   * - xfig:       Fig file
   * - hp7470:     HP 7470 Plotter File (HPGL Cartridge, Small Plotter)
   * - hp7580:     HP 7580 Plotter File (Large Plotter)
   * - lj_hpgl:    HP Laserjet III, HPGL emulation mode
   * - pbm:        PDB (PPM) Driver
   * - png:        PNG file
   * - jpeg:       JPEG file
   * - null:       Null device
   * - tkwin:      New tk driver
   * - mem:        User-supplied memory device
   * - gif:        GIF file
   * - svg:        Scalable Vector Graphics (SVG 1.1)
   *
   * @remarks: this is set with plot_initialize().
   */
  
  const char *device;
  int stream;	 /**< The plots current stream ID */
  int *tot_streams;  /**< Keeps strack of total streams */
  
  const char *xlabel;	 /**< Label for the x-axis */
  const char *ylabel;	 /**< Label for the y-axis */
  const char *title;	/**< Plot title */
  plot_color_type label_color;  /**< Color for the labels */
  
  int 	 height;		/**< The height of your plot window */
  int 	 width;         	/**< The width of your plot window */
  double xmin,xmax,ymin,ymax;   /**< Ranges for plot. */ 
  bool   __use_autorange;       
  plot_range_type * range;      /**< Range instance (not in use yet). */
};



int plot_get_stream(plot_type * item)
{
    assert(item != NULL);
    return item->stream;
}


list_type *plot_get_datasets(plot_type * item)
{
    assert(item != NULL);
    return item->datasets;
}


/**
 * @return Returns a new plot_type pointer.
 * @brief Create a new plot_type
 *
 * Create a new plot - allocates the memory.
 */
plot_type *plot_alloc()
{
  plot_type *item;
  
  item = util_malloc(sizeof *item , __func__);
  item->stream = -1;
  item->xlabel = NULL;
  item->ylabel = NULL;
  item->title = NULL;
  item->datasets = list_alloc();
  item->height 	  = DEFAULT_HEIGHT;
  item->width  	  = DEFAULT_WIDTH;
  item->__use_autorange = true;
  item->range           = plot_range_alloc();
  return item;
}



/**
 * @brief Setup window size.
 * @param item your current plot
 * @param width the width of your window.
 * @param height the height of you window.
 * 
 * Sets up your window geometry. This has to be set before initializing the window, else default size is set.
 */
void plot_set_window_size(plot_type * item, int width, int height)
{
    assert(item != NULL);
    item->width = width;
    item->height = height;
}


/**
 * @brief Initialize a new plot
 * @param item your current plot
 * @param dev the output device
 * @param filename the output filename 
 * @param w define it we are plotting to a canvas or normal
 * 
 * This function has to be called before you set any other information 
 * on the plot_type *item. This is because plinit() gets called inside this function.
 */
void plot_initialize(plot_type * item, const char *dev, const char *filename) {
  static int output_stream = 0;
  static int tot_streams = 0;
  pthread_mutex_t update_lock;
  char buf[16];
  
  assert(item != NULL);
  item->stream = output_stream;
  item->tot_streams = &tot_streams;
  item->device = dev;
  item->filename = filename;
  plsstrm(item->stream);
  
  if (dev) {
    if (strcmp(item->device, "xwin"))
      plsfnam(item->filename);
  }
  if (dev)
    plsdev(item->device);

  plscol0(WHITE, 255, 255, 255);
  plscol0(BLACK, 0, 0, 0);
  snprintf(buf, sizeof(buf), "%dx%d", item->width, item->height);
  plsetopt("geometry", buf);
  plfontld(0);
  plinit();
  
  /* Make this threadsafe */
  pthread_mutex_init(&update_lock, NULL);
  pthread_mutex_lock(&update_lock);
  /* Use another input stream next time */
  output_stream++;
  tot_streams++;
  pthread_mutex_unlock(&update_lock);
}



/**
 * @brief Free all the datasets in a plot item
 * @param item your current plot
 * 
 * Using this function is optional, the datasets WILL be freed  when plot_free() is called anyway.
 */
void plot_free_all_datasets(plot_type * item)
{
    list_node_type *node, *next_node;

    assert(item->datasets != NULL);
    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
	list_del_node(item->datasets, node);
	plot_dataset_free(tmp);
	node = next_node;
    }

    list_free(item->datasets);
    item->datasets = NULL;
}

/**
 * @brief Free your plot item
 * @param item your current plot
 * 
 * Use this function to free your allocated memory from plot_alloc().
 */
void plot_free(plot_type * item)
{
    pthread_mutex_t update_lock;

    assert(item != NULL);
    plot_free_all_datasets(item);

    plsstrm(item->stream);
    plend1();

    pthread_mutex_init(&update_lock, NULL);
    pthread_mutex_lock(&update_lock);
    --*item->tot_streams;
    if (!*item->tot_streams) 
      plend();
    
    pthread_mutex_unlock(&update_lock);
    plot_range_free(item->range);
    util_safe_free(item);
}


static void plot_set_range__(plot_type * plot) {
  if (plot->__use_autorange)
    plot_get_extrema(plot , &plot->xmin , &plot->xmax , &plot->ymin , &plot->ymax);
  
  
  plsstrm(plot->stream);
  
  plvsta();
  plwind(plot->xmin, plot->xmax, plot->ymin, plot->ymax);
  plcol0(BLACK);
  plschr(0, LABEL_FONTSIZE);
  plbox("bcnst", 0.0 , 0 , "bcnstv" , 0.0 , 0);

  plschr(0, LABEL_FONTSIZE);
  plcol0(plot->label_color);
  pllab(plot->xlabel, plot->ylabel, plot->title);
}



/**
 * @brief Do the actual plotting
 * @param item your current plot
 * 
 * After adding the datasets, setting viewport and labels you are ready to do the plotting.
 */
void plot_data(plot_type * item)
{
    list_node_type *node, *next_node;
    
    plot_set_range__(item);

    assert(item != NULL);
    plsstrm(item->stream);
    node = list_get_head(item->datasets);
    while (node != NULL) {
	plot_dataset_type *tmp;

	next_node = list_node_get_next(node);
	tmp = list_node_value_ptr(node);
	plot_dataset(item, tmp);
	node = next_node;
    }
}


void plot_errorbar_data(plot_type * item)
{
    list_node_type *node, *next_node;
    PLFLT *ymin = NULL;
    PLFLT *ymax = NULL;
    int tmp_len = 0;
    int i;
    plot_dataset_type *ref = NULL;

    assert(item != NULL);
    plsstrm(item->stream);
    ref = plot_dataset_get_prominent(item, &tmp_len);

    for (i = 0; i <= tmp_len; i++) {
	PLFLT max = 0;
	PLFLT min = 0;
	bool flag = false;

	node = list_get_head(item->datasets);
	while (node != NULL) {
	    plot_dataset_type *tmp;
	    PLFLT *y;

	    next_node = list_node_get_next(node);
	    tmp = list_node_value_ptr(node);
	    y = plot_dataset_get_vector_y(tmp);
	    if (!flag) {
		max = y[i];
		min = y[i];
		flag = true;
	    }
	    if (y[i] > max)
		max = y[i];
	    if (y[i] < min)
		min = y[i];
	    node = next_node;
	}

	assert(i >= 0);
	ymin = realloc(ymin, sizeof(PLFLT) * (i + 1));
	ymax = realloc(ymax, sizeof(PLFLT) * (i + 1));
	ymin[i] = min;
	ymax[i] = max;
    }

    assert(tmp_len > 0);
    assert(ymax != NULL && ymax != NULL);
    plerry(tmp_len, plot_dataset_get_vector_x(ref), ymin, ymax);
    util_safe_free(ymin);
    util_safe_free(ymax);
}


/*void plot_std_data(plot_type * item, bool mean)
{
    list_node_type *node, *next_node;
    int tmp_len = 0;
    int i;
    plot_dataset_type *ref = NULL;
    PLFLT *ymin = NULL;
    PLFLT *ymax = NULL;
    PLFLT *vec_mean = NULL;

    assert(item != NULL);
    plsstrm(item->stream);
    ref = plot_dataset_get_prominent(item, &tmp_len);

    for (i = 0; i <= tmp_len; i++) {
	int j = -1;
	PLFLT *vec = 0;
	PLFLT rms;
	PLFLT vec_sum = 0;

	node = list_get_head(item->datasets);
	while (node != NULL) {
	    plot_dataset_type *tmp;
	    PLFLT *y;

	    next_node = list_node_get_next(node);
	    tmp = list_node_value_ptr(node);
	    y = plot_dataset_get_vector_y(tmp);
	    j++;
	    vec = realloc(vec, sizeof(PLFLT) * (j + 1));
	    vec[j] = y[i];
	    vec_sum += vec[j];

	    node = next_node;
	}
	rms = plot_util_calc_rms(vec, j);
	assert(i >= 0);
	ymin = realloc(ymin, sizeof(PLFLT) * (i + 1));
	ymax = realloc(ymax, sizeof(PLFLT) * (i + 1));
	if (mean) {
	    vec_mean = realloc(vec_mean, sizeof(PLFLT) * (i + 1));
	    vec_mean[i] = vec_sum / (PLFLT) (j + 1);
	}
	ymin[i] = (vec_sum / (PLFLT) (j + 1)) - rms;
	ymax[i] = (vec_sum / (PLFLT) (j + 1)) + rms;;
	util_safe_free(vec);
    }

    assert(tmp_len > 0);
    assert(ymax != NULL && ymax != NULL);
    plerry(tmp_len, plot_dataset_get_vector_x(ref), ymin, ymax);
    if (mean) {
	plot_dataset_type *d;

	d = plot_dataset_alloc( false );
	plot_dataset_set_data(d, plot_dataset_get_vector_x(ref), vec_mean,
			      tmp_len, RED, LINE);
	plot_dataset(item, d);
	plot_dataset_free(d);
	util_safe_free(vec_mean);
    }
    util_safe_free(ymin);
    util_safe_free(ymax);
}
*/


/**
 * @brief Set labels
 * @param item your current plot
 * @param xlabel label for the x-axis
 * @param ylabel label for the y-axis
 * @param title title for the plot
 * @param color the color for your labels
 * 
 * Set the labels for your plot, do this before setting the viewport.
 */
void
plot_set_labels(plot_type * item, const char *xlabel, const char *ylabel,
		const char *title, plot_color_type color)
{
    assert(item != NULL);
    item->xlabel = xlabel;
    item->ylabel = ylabel;
    item->title = title;
    item->label_color = color;
}



/* 
   This is the low-level function setting the range of the plot.
*/
   


void plot_set_manual_range(plot_type * plot , double xmin , double xmax , double ymin , double ymax) {
  plot->__use_autorange = false;
  plot->xmin = xmin;
  plot->xmax = xmax;
  plot->ymin = ymin;
  plot->ymax = ymax;
}


/**
 * @brief Setup viewport
 * @param item your current plot
 * @param xmin minimum value for the x-axis
 * @param xmax maximum value for the x-axis
 * @param ymin minimum value for the y-axis
 * @param ymax maximum value for the y-axis
 * 
 * Sets up your viewport, defining and the axis, setting up fonts and colors.
 */
void plot_set_viewport(plot_type * item) {
  assert(item != NULL);
  plsstrm(item->stream);
  pladv(0);
  plcol0(BLACK);
  plschr(0, LABEL_FONTSIZE);
  //plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);
  
  if (!item->xlabel || !item->ylabel || !item->title) {
    fprintf(stderr,
	    "ERROR ID[%d]: you need to set lables before setting the viewport!\n",
	    item->stream);
  } else {
    plschr(0, LABEL_FONTSIZE);
    plcol0(item->label_color);
    //pllab(item->xlabel, item->ylabel, item->title);
  }
}



/**
 * @brief Get extrema values
 * @param item your current plot
 * @param x_max pointer to the new x maximum
 * @param y_max pointer to the new y maximum
 * @param x_min pointer to the new x minimum
 * @param y_min pointer to the new y minimum
 * 
 * Find the extrema values in the plot item, checks all added datasets.
 */

 void plot_get_extrema(plot_type * item, double *x_min, double *x_max,double *y_min, double *y_max) {
  bool first_pass = true;
  list_node_type *node, *next_node;
  assert(item != NULL);
  node = list_get_head(plot_get_datasets(item));
  while (node != NULL) {
    plot_dataset_type *tmp;
    next_node = list_node_get_next(node);
    tmp = list_node_value_ptr(node);
    
    plot_dataset_update_range(tmp , first_pass , x_min , x_max , y_min , y_max);
    node = next_node;
    first_pass = false;
  }
}


