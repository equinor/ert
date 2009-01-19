#include <stdbool.h>
#include <plot.h>
#include <plot_dataset.h>
#include <plot_range.h>
#include <plot_const.h>

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

/**
 * @brief Contains information about a plotting window.
 */

struct plot_struct {
  const char *filename; /**< Filename for the plot */    
  int                  alloc_size;  	/* The size of the dataset vector - can in general be larger than size. */
  int                  size;        	/* The number of datasets. */
  plot_dataset_type ** datasets;    	/* Pointers to datasets. */
  bool                 is_histogram;    /* If this is true it can only contain histogram datasets. */
  
  
  const char *device;
  int stream;	     /**< The plots current stream ID */
  int *tot_streams;  /**< Keeps strack of total streams */
  
  char *xlabel;	 /**< Label for the x-axis */
  char *ylabel;	 /**< Label for the y-axis */
  char *title;	/**< Plot title */

  plot_color_type label_color;  /**< Color for the labels */
  plot_color_type box_color;    /**< Color for the axis / box surrounding the plot. */
  double label_font_size;       /**< Scale factor for label font size. */ 
  int 	 height;		/**< The height of your plot window */
  int 	 width;         	/**< The width of your plot window */

  bool   __use_autorange;       
  plot_range_type * range;       /**< Range instance */
};



int plot_get_stream(plot_type * plot) {
  return plot->stream;
}


static void plot_realloc_datasets(plot_type * plot, int new_alloc_size) {
  int i;
  plot->datasets = util_realloc(plot->datasets , new_alloc_size * sizeof * plot->datasets, __func__);
  plot->alloc_size = new_alloc_size;
  for (i = plot->size; i < plot->alloc_size; i++)
    plot->datasets[i] = NULL;
}


/**
 * @brief Setup window size.
 * @param plot your current plot
 * @param width the width of your window.
 * @param height the height of you window.
 * 
 * Sets up your window geometry. This has to be set before initializing the window, else default size is set.
 */
void plot_set_window_size(plot_type * plot, int width, int height)
{
  plot->width  = width;
  plot->height = height;
}



/**
 * @return Returns a new plot_type pointer.
 * @brief Create a new plot_type
 *
 * Create a new plot - allocates the memory.
 */
plot_type *plot_alloc()
{
  plot_type *plot;
  
  plot = util_malloc(sizeof *plot , __func__);
  plot->stream 	  	= -1;
  plot->xlabel 	  	= NULL;
  plot->ylabel 	  	= NULL;
  plot->title  	  	= NULL;
  plot->__use_autorange = true;
  plot->datasets     	= NULL;
  plot->size         	= 0;
  plot->alloc_size   	= 0; 
  plot->is_histogram 	= false;
  
  plot->range      = plot_range_alloc();
  plot_set_window_size(plot , PLOT_DEFAULT_WIDTH , PLOT_DEFAULT_HEIGHT);
  plot_set_box_color(plot , PLOT_DEFAULT_BOX_COLOR);
  plot_set_label_color(plot , PLOT_DEFAULT_LABEL_COLOR);
  plot_set_label_fontsize(plot , 1.0);
  plot_set_labels(plot , "" , "" , ""); /* Initializeing with empty labels. */
  return plot;
}


/**
   Allocates a new dataset, and attaches it to the plot. When adding
   data to the datset, setting attributes+++ you should use
   plot_dataset_xxx functions with the return value from this
   function.
*/

plot_dataset_type * plot_alloc_new_dataset(plot_type * plot , plot_data_type data_type, bool shared_data) {
  if (data_type == plot_hist) {
    if (plot->size > 0)
      util_abort("%s: sorry - when using histograms you can *only* have one dataset\n",__func__);
    plot->is_histogram = true;
  } else if (plot->is_histogram)
    util_abort("%s: sorry - when using histograms you can *only* have one dataset\n",__func__);
  
  {
    plot_dataset_type * dataset = plot_dataset_alloc(data_type, shared_data);
    if (plot->size == plot->alloc_size)
      plot_realloc_datasets(plot , 2*(plot->size + 1));
    plot->datasets[plot->size] = dataset;
    plot->size++;
    return dataset;
  }
}  





/**
 * @brief Initialize a new plot
 * @param plot your current plot
 * @param dev the output device
 * @param filename the output filename 
 * @param w define it we are plotting to a canvas or normal
 * 
 * This function has to be called before you set any other information
 * on the plot_type *plot. This is because plinit() is called inside
 * this function.
 */

void plot_initialize(plot_type * plot, const char *dev, const char *filename) {
  static int output_stream = 0;
  static int tot_streams   = 0;

  
  plot->stream      = output_stream;
  plot->tot_streams = &tot_streams;
  plot->device      = dev;
  plot->filename    = filename;
  plsstrm(plot->stream);
  
  if (dev) {
    if (strcmp(plot->device, "xwin"))
      plsfnam(plot->filename);
  }
  if (dev)
    plsdev(plot->device);

  /** This color initialization must be here - do not really understand what for. */
  plscol0(WHITE, 255, 255, 255);
  plscol0(BLACK, 0, 0, 0);
  
  {
    char * geometry = util_alloc_sprintf("%dx%d", plot->width, plot->height);
    plsetopt("geometry", geometry);
    free(geometry);
  }
  
  plfontld(0);
  plinit();
  
  /* Use another input stream next time */
  output_stream++;
  tot_streams++;
}



static void plot_free_all_datasets(plot_type * plot)
{
  int i;
  for (i=0; i < plot->size; i++)
    plot_dataset_free(plot->datasets[i]);
  util_safe_free(plot->datasets);
  plot->datasets   = NULL;
  plot->alloc_size = 0;
  plot->size       = 0;
}

/**
 * @brief Free your plot plot
 * @param plot your current plot
 * 
 * Use this function to free your allocated memory from plot_alloc().
 */
void plot_free(plot_type * plot)
{
  
  plot_free_all_datasets(plot);
  
  plsstrm(plot->stream);
  plend1();
  
  --*plot->tot_streams;
  if (!*plot->tot_streams) 
    plend();
  
  plot_range_free(plot->range);
  util_safe_free(plot);
}


static void plot_set_range__(plot_type * plot, double *x1 , double *x2 , double *y1 ,double *y2) {
  if (plot->__use_autorange)
    plot_get_extrema(plot , plot->range);

  plot_range_apply(plot->range , x1 , x2 , y1 , y2);
}



/**
   This function does the actual plotting. Observe the following design principle:

    * All the toolkit dependant functions (i.e. plxxxx in the case of
      plplot) should be assembled here in this function. This can
      somethimes be a bit awkward, but should simplify the process of
      porting to another toolkit for plotting.
      
    * The exception to this rule is the function plot_dataset_draw()
      which will invoke toolkit spesific functions for drawing
      lines/point/errorbars/...

*/

void plot_data(plot_type * plot)
{
  int iplot;
  
  plsstrm(plot->stream);  

  if (plot->is_histogram) {
    plot_set_range__(plot , NULL , NULL , NULL , NULL);
    for (iplot = 0; iplot < plot->size; iplot++) 
      plot_dataset_draw(plot->stream , plot->datasets[iplot] , plot->range);
    
    return;
  } 


  pladv(0);  /* And what does this do ... */
  plvsta();
  {
    double x1,x2,y1,y2;
    plot_set_range__(plot , &x1 , &x2 , &y1 , &y2);  
    plwind(x1,x2,y1,y2);
  }
  
  plcol0(plot->label_color);
  plschr(0, plot->label_font_size * PLOT_DEFAULT_LABEL_FONTSIZE);
  plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);
  
  if (!plot->xlabel || !plot->ylabel || !plot->title) 
    fprintf(stderr, "ERROR ID[%d]: you need to set lables before setting the viewport!\n",plot->stream);
  else {
    plcol0(plot->label_color);
    plschr(0, plot->label_font_size * PLOT_DEFAULT_LABEL_FONTSIZE);
    pllab(plot->xlabel, plot->ylabel, plot->title);
  }
  
  for (iplot = 0; iplot < plot->size; iplot++) 
    plot_dataset_draw(plot->stream , plot->datasets[iplot] , plot->range);
}


void plot_set_xlabel(plot_type * plot , const char * xlabel) {
  plot->xlabel = util_realloc_string_copy(plot->xlabel , xlabel);
}

void plot_set_ylabel(plot_type * plot , const char * ylabel) {
  plot->ylabel = util_realloc_string_copy(plot->ylabel , ylabel);
}

void plot_set_title(plot_type * plot , const char * title) {
  plot->title = util_realloc_string_copy(plot->title , title);
}


void plot_set_labels(plot_type * plot, const char *xlabel, const char *ylabel, const char *title)
{
  plot_set_xlabel(plot , xlabel);
  plot_set_ylabel(plot , ylabel);
  plot_set_title(plot , title);
}



/* 
   This is the low-level function setting the range of the plot.
*/
   


void plot_set_manual_range(plot_type * plot , double xmin , double xmax , double ymin , double ymax) {
  plot->__use_autorange = false;
  plot_range_set_xmin(plot->range , xmin);
  plot_range_set_xmax(plot->range , xmax);
  plot_range_set_ymin(plot->range , ymin);
  plot_range_set_ymax(plot->range , ymax);
}


void plot_set_left_padding(plot_type * plot , double value) {
  plot_range_set_left_padding(plot->range, value);
}

void plot_set_right_padding(plot_type * plot , double value) {
  plot_range_set_right_padding(plot->range , value);
}

void plot_set_top_padding(plot_type * plot , double value) {
  plot_range_set_top_padding(plot->range , value);
}

void plot_set_bottom_padding(plot_type * plot , double value) {
  plot_range_set_bottom_padding(plot->range , value);
}

/*****************************************************************/

void plot_invert_x_axis(plot_type * plot) {
  plot_range_invert_x_axis(plot->range , true);
}

void plot_invert_y_axis(plot_type * plot) {
  plot_range_invert_y_axis(plot->range , true);
}

/*****************************************************************/

void plot_set_label_color(plot_type * plot , plot_color_type label_color) {
  plot->label_color = label_color;
}


void plot_set_box_color(plot_type * plot , plot_color_type box_color) {
  plot->box_color = box_color;
}

void plot_set_label_fontsize(plot_type * plot , double label_font_size_scale) {
  plot->label_font_size = label_font_size_scale;
}


/**
 * @brief Get extrema values
 * @param plot your current plot
 * @param x_max pointer to the new x maximum
 * @param y_max pointer to the new y maximum
 * @param x_min pointer to the new x minimum
 * @param y_min pointer to the new y minimum
 * 
 * Find the extrema values in the plot plot, checks all added datasets.
 */

void plot_get_extrema(plot_type * plot, plot_range_type * range) {
  bool first_pass = true;
  int iplot;
  for (iplot = 0; iplot < plot->size; iplot++) {
    plot_dataset_update_range(plot->datasets[iplot] , first_pass , range);
    first_pass = false;
  }
}


