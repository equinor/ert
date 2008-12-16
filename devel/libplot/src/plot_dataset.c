#include <util.h>
#include <plot.h>
#include <plot_dataset.h>
#include <plot_const.h>
#include <plot_range.h>
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
  plot_data_type  data_type;       /**< One of the types: plot_xy, plot_xy1y2, plot_x1x2y , plot_xline , plot_yline */      
  
  bool    shared_data;     /**< Whether this instance owns x,y,x1,... or just holds a reference. */
  int alloc_size;          /**< The allocated size of x,y (std) - will be 0 if shared_data = true. */
  int size;	   	   /**< Length of the vectors defining the plot */




  /* All of these are manipulated through obvious functions - but the default should be sensible. */
  plot_style_type      style;        /**< The graph style: line|points|line_points */
  plot_color_type      line_color;   /**< The color for the line part of the plot. */
  plot_color_type      point_color;  /**< The color for the points in the plot. */
  plot_symbol_type     symbol_type;  /**< The type of symbol. */
  plot_line_style_type line_style;   /**< The style for lines. */
  double               symbol_size;  /**< Scale factor for symbol size : starts with PLOT_DEFAULT_SYMBOL_SIZE */
  double               line_width;   /**< Scale factor for line width  : starts with PLOT_DEFAULT_LINE_WIDTH */
};

/*****************************************************************/
/* Set - functions for the style variabels of the dataset. */
void plot_dataset_set_style(plot_dataset_type * dataset , plot_style_type style) {
  dataset->style = style;
}


void plot_dataset_set_line_color(plot_dataset_type * dataset , plot_color_type line_color) {
  dataset->line_color = line_color;
}


void plot_dataset_set_point_color(plot_dataset_type * dataset , plot_color_type point_color) {
  dataset->point_color = point_color;
}


void plot_dataset_set_line_style(plot_dataset_type * dataset , plot_line_style_type line_style) {
  dataset->line_style = line_style;
}

void plot_dataset_set_symbol_size(plot_dataset_type * dataset , double symbol_size) {
  dataset->symbol_size = symbol_size;
}

void plot_dataset_set_line_width(plot_dataset_type * dataset , double line_width) {
  dataset->symbol_size = line_width;
}

/*****************************************************************/

double * plot_dataset_get_vector_x(const plot_dataset_type * d)  { return d->x; }
double * plot_dataset_get_vector_y(const plot_dataset_type * d)  { return d->y; }
double * plot_dataset_get_vector_x1(const plot_dataset_type * d) { return d->x1; }
double * plot_dataset_get_vector_y1(const plot_dataset_type * d) { return d->y1; }
double * plot_dataset_get_vector_x2(const plot_dataset_type * d) { return d->x2; }
double * plot_dataset_get_vector_y2(const plot_dataset_type * d) { return d->y2; }


static void plot_dataset_assert_type(const plot_dataset_type * d, plot_data_type type) {
  if (d->data_type != type)
    util_abort("%s: assert failed - wrong plot type \n",__func__);
}



static int  __make_data_mask(plot_data_type data_type) {
  int mask = 0;
  switch (data_type) {
  case(plot_xy):
    mask = plot_data_x + plot_data_y;
    break;
  case(plot_xy1y2):
    mask = plot_data_x + plot_data_y1 + plot_data_y2;
    break;
  case(plot_x1x2y):
    mask = plot_data_x1 + plot_data_x2 + plot_data_y;
    break;
  case(plot_xline):
    mask = plot_data_x;
    break;
  case(plot_yline):
    mask = plot_data_y;
    break;
  default:
    util_abort("%s: unrecognized value: %d \n",__func__ , data_type);
  }
  return mask;
}





/**
 * @return Returns a new plot_dataset_type pointer.
 * @brief Create a new plot_dataset_type
 *
 * Create a new dataset - allocates the memory.
 */
plot_dataset_type *plot_dataset_alloc(plot_data_type data_type , bool shared_data)
{
  plot_dataset_type *d;
  
  d = util_malloc(sizeof *d , __func__);
  d->data_type     = data_type;
  d->data_mask     = __make_data_mask(data_type);
  d->x   	   = NULL;
  d->y   	   = NULL;
  d->x1 	   = NULL;
  d->x2 	   = NULL;
  d->y1 	   = NULL;
  d->y2 	   = NULL;
  d->size          = 0;
  d->alloc_size    = 0;
  d->shared_data   = shared_data;


  /******************************************************************/
  /* Defaults - should be installed in some way? */
  d->style       = LINE;
  d->line_color  = BLUE;
  d->point_color = BLUE;
  d->symbol_type = 17;
  d->line_style  = solid_line;
  d->symbol_size = 1.0;  /* Scale factor */
  d->line_width  = 1.0;  /* Scale factor */
  return d;
}


/**
   This function frees all the memory related to a dataset - normally
   called automatically from plot_free().
*/
void plot_dataset_free(plot_dataset_type * d)
{
  if (!d->shared_data) {
    util_safe_free(d->x);
    util_safe_free(d->x1);
    util_safe_free(d->x2);
    util_safe_free(d->y);
    util_safe_free(d->y1);
    util_safe_free(d->y2);
  }
  free(d);
}

/*****************************************************************/
/** Here comes functions for setting the data for the dataset. */


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


static void plot_dataset_set_shared__(plot_dataset_type * d , int size , double *x , double *y, double *y1 , double *y2 , double *x1, double *x2) {
  if (d->shared_data) {
    d->x    = x;
    d->y    = y;
    d->x1   = x1;
    d->x2   = x2;
    d->y1   = y1;
    d->y2   = y2;
    d->size = size;
  } else
    util_abort("%s ... \n");
}

/*****************************************************************/
/* Here comes the exported functions - they all have _xy, _xy1y2,
   _xline ... suffix. */


void plot_dataset_append_vector_xy(plot_dataset_type *d , int size, const double * x , const double *y) {
  plot_dataset_assert_type(d , plot_xy);
  plot_dataset_append_vector__(d , size , x , y , NULL , NULL , NULL , NULL);
}


void plot_dataset_append_point_xy(plot_dataset_type *d , double x , double y) {
  plot_dataset_append_vector_xy(d , 1 , &x , &y);
}


void plot_dataset_set_shared_xy(plot_dataset_type *d , int size, double *x, double *y) {
  plot_dataset_assert_type(d , plot_xy);
  plot_dataset_set_shared__(d , size , x , y , NULL , NULL , NULL , NULL);
}


/*----*/


void plot_dataset_append_vector_xy1y2(plot_dataset_type *d , int size, const double * x , const double *y1 , const double *y2) {
  plot_dataset_assert_type(d , plot_xy1y2);
  plot_dataset_append_vector__(d , size , x , NULL , y1 , y2 , NULL , NULL);
}


void plot_dataset_append_point_xy1y2(plot_dataset_type *d , double x , double y1 , double y2) {
  plot_dataset_append_vector_xy1y2(d , 1 , &x , &y1, &y2);
}


void plot_dataset_set_shared_xy1y2(plot_dataset_type *d , int size, double *x, double *y1 , double *y2) {
  plot_dataset_assert_type(d , plot_xy1y2);
  plot_dataset_set_shared__(d , size , x , NULL , y1 , y2 , NULL , NULL);
}

/*----*/

void plot_dataset_append_vector_x1x2y(plot_dataset_type *d , int size, const double * x1 , const double *x2 , const double *y) {
  plot_dataset_assert_type(d , plot_x1x2y);
  plot_dataset_append_vector__(d , size , NULL , y , NULL , NULL , x1 , x2);
}


void plot_dataset_append_point_x1x2y(plot_dataset_type *d , double x1 , double x2 , double y) {
  plot_dataset_append_vector_x1x2y(d , 1 , &x1 , &x2 , &y);
}


void plot_dataset_set_shared_x1x2y(plot_dataset_type *d , int size, double *x1, double *x2 , double *y) {
  plot_dataset_assert_type(d , plot_x1x2y);
  plot_dataset_set_shared__(d , size , NULL , y , NULL , NULL , x1 , x2);
}

/*----*/

void plot_dataset_append_vector_xline(plot_dataset_type *d , int size, const double * x) {
  plot_dataset_assert_type(d , plot_xline);
  plot_dataset_append_vector__(d , size , x , NULL , NULL , NULL , NULL , NULL);
}


void plot_dataset_append_point_xline(plot_dataset_type *d , double x) {
  plot_dataset_append_vector_xline(d , 1 , &x);
}


void plot_dataset_set_shared_xline(plot_dataset_type *d , int size, double *x) {
  plot_dataset_assert_type(d , plot_xline);
  plot_dataset_set_shared__(d , size , x , NULL , NULL , NULL , NULL , NULL);
}


/*----*/

void plot_dataset_append_vector_yline(plot_dataset_type *d , int size, const double * y) {
  plot_dataset_assert_type(d , plot_yline);
  plot_dataset_append_vector__(d , size , NULL , y , NULL , NULL , NULL , NULL );
}


void plot_dataset_append_point_yline(plot_dataset_type *d , double y) {
  plot_dataset_append_vector_yline(d , 1 , &y);
}


void plot_dataset_set_shared_yline(plot_dataset_type *d , int size, double *y) {
  plot_dataset_assert_type(d , plot_yline);
  plot_dataset_set_shared__(d , size , NULL , y , NULL , NULL , NULL , NULL);
}

/*****************************************************************/





void plot_dataset_draw(int stream, plot_dataset_type * d , const plot_range_type * range) {
  plsstrm(stream);
  pllsty(d->line_style);                                  /* Setting solid/dashed/... */
  plwid(d->line_width * PLOT_DEFAULT_LINE_WIDTH);         /* Setting line width.*/
  plcol0(d->line_color);                                  /* Setting line color. */
  plssym(0 , d->symbol_size * PLOT_DEFAULT_SYMBOL_SIZE);  /* Setting the size for the symbols. */

  
  switch (d->data_type) {
  case(plot_xy):
    /** Starting with the line */    
    if (d->style == LINE || d->style == LINE_POINTS) 
      plline(d->size , d->x , d->y);

    if (d->style == POINT || d->style == LINE_POINTS) {
      plcol0(d->point_color);       /* Setting the color */
      plpoin(d->size , d->x , d->y , d->symbol_type);
    }
    break;
  case(plot_xy1y2):
    plerry(d->size , d->x , d->y1 , d->y2);
    break;
  case(plot_x1x2y):
    plerrx(d->size , d->x1 , d->x2 , d->y);
    break;
  case(plot_yline):
    {
      double x[2] = {plot_range_get_xmin(range) , plot_range_get_xmax(range)};
      double y[2];
      
      for (int i=0; i < d->size; i++) {
	y[0] = d->y[i];
	y[1] = d->y[i];
	plline(2 , x , y);
      }
    }
    break;
  case(plot_xline):
    {
      double y[2] = {plot_range_get_ymin(range) , plot_range_get_ymax(range)};
      double x[2];
      
      for (int i=0; i < d->size; i++) {
	x[0] = d->x[i];
	x[1] = d->x[i];
	plline(2 , x , y);
      }
    }
    break;
  default:
    util_abort("%s: internal error ... \n",__func__);
    break;
  }
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
void plot_dataset_update_range(plot_dataset_type * d, bool first_pass , plot_range_type * range) {
  double tmp_x_max = plot_range_safe_get_xmax(range);
  double tmp_y_max = plot_range_safe_get_ymax(range);
  double tmp_x_min = plot_range_safe_get_xmin(range);
  double tmp_y_min = plot_range_safe_get_ymin(range);

  int i;
  double *x1 , *x2, *y1 , *y2;


  x1 = NULL;
  x2 = NULL;
  y1 = NULL;
  y2 = NULL;
  if (d->data_mask & plot_data_x)  {x1 = d->x;  x2 = d->x; }
  if (d->data_mask & plot_data_x1)  x1 = d->x1;
  if (d->data_mask & plot_data_x2)  x2 = d->x2;

  if (d->data_mask & plot_data_y)  {y1 = d->y;  y2 = d->y; }
  if (d->data_mask & plot_data_y1)  y1 = d->y1;
  if (d->data_mask & plot_data_y2)  y2 = d->y2;

  if (x1 != NULL) {
    if (first_pass) {
      tmp_x_min = x1[0];
      tmp_x_max = x2[0];
    }
      
    for (i=0; i < d->size; i++) {
      if (x1[i] < tmp_x_min)
	tmp_x_min = x1[i];

      if (x2[i] > tmp_x_max)
	tmp_x_max = x2[i];
    }
  }


  if (y1 != NULL) {
    if (first_pass) {
      tmp_y_min = y1[0];
      tmp_y_max = y2[0];
    }
      
    for (i=0; i < d->size; i++) {
      if (y1[i] < tmp_y_min)
	tmp_y_min = y1[i];

      if (y2[i] > tmp_y_max)
	tmp_y_max = y2[i];
    }
  }


  plot_range_set_xmax(range , tmp_x_max);
  plot_range_set_ymax(range , tmp_y_max);
  plot_range_set_xmin(range , tmp_x_min);
  plot_range_set_ymin(range , tmp_y_min);
}


