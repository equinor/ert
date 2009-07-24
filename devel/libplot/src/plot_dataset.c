#include <string.h>
#include <util.h>
#include <plplot/plplot.h>
#include <plplot/plplotP.h>
#include <math.h>
#include <double_vector.h>
#include <bool_vector.h>
#include <plot_const.h>
#include <plot_range.h>
#include <plot_dataset.h>
#include <plot_driver.h>




#define PLOT_DATASET_TYPE_ID 661409

/**
 * @brief Contains information about a dataset.
 */
struct plot_dataset_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                *label;             /* Label for the dataset - used as hash key in the plot instance. */ 
  double_vector_type  *x; 	   	   /**< Vector containing x-axis data */
  double_vector_type  *y; 	   	   /**< Vector containing y-axis data */
  double_vector_type  *y1;              
  double_vector_type  *y2;
  double_vector_type  *x1;
  double_vector_type  *x2;
  int     	      data_mask;       /**< An integer value written as a sum of values from the enum plot_data_types - says which type of data (x/y/...) is supplied. */
  plot_data_type      data_type;       /**< One of the types: plot_xy, plot_xy1y2, plot_x1x2y , plot_xline , plot_yline */      

  plot_style_type       style;        /**< The graph style: line|points|line_points */

  line_attribute_type   line_attr;
  point_attribute_type  point_attr;
};



/*****************************************************************/
/* Set - functions for the style variabels of the dataset.       */

int plot_dataset_get_size( const plot_dataset_type * dataset ) {

  if (dataset->data_mask & PLOT_DATA_X)  return double_vector_size( dataset->x );
  if (dataset->data_mask & PLOT_DATA_Y)  return double_vector_size( dataset->y );
  if (dataset->data_mask & PLOT_DATA_X1) return double_vector_size( dataset->x1 );
  if (dataset->data_mask & PLOT_DATA_Y1) return double_vector_size( dataset->y1 );
  if (dataset->data_mask & PLOT_DATA_X2) return double_vector_size( dataset->x2 );
  if (dataset->data_mask & PLOT_DATA_Y2) return double_vector_size( dataset->y2 );
  
  util_abort("%s: internal error\n",__func__);
  return -1;
}

void plot_dataset_set_style(plot_dataset_type * dataset , plot_style_type style) {
  dataset->style = style;
}


void plot_dataset_set_line_color(plot_dataset_type * dataset , plot_color_type line_color) {
  dataset->line_attr.line_color = line_color;
}


void plot_dataset_set_point_color(plot_dataset_type * dataset , plot_color_type point_color) {
  dataset->point_attr.point_color = point_color;
}


void plot_dataset_set_line_style(plot_dataset_type * dataset , plot_line_style_type line_style) {
  dataset->line_attr.line_style = line_style;
}

void plot_dataset_set_symbol_size(plot_dataset_type * dataset , double symbol_size) {
  dataset->point_attr.symbol_size = symbol_size;
}

void plot_dataset_set_line_width(plot_dataset_type * dataset , double line_width) {
  dataset->line_attr.line_width = line_width;
}

void plot_dataset_set_symbol_type(plot_dataset_type * dataset, plot_symbol_type symbol_type) {
  dataset->point_attr.symbol_type = symbol_type;
}


/*****************************************************************/

//double * plot_dataset_get_vector_x(const plot_dataset_type * d)  { return d->x; }
//double * plot_dataset_get_vector_y(const plot_dataset_type * d)  { return d->y; }
//double * plot_dataset_get_vector_x1(const plot_dataset_type * d) { return d->x1; }
//double * plot_dataset_get_vector_y1(const plot_dataset_type * d) { return d->y1; }
//double * plot_dataset_get_vector_x2(const plot_dataset_type * d) { return d->x2; }
//double * plot_dataset_get_vector_y2(const plot_dataset_type * d) { return d->y2; }


static void plot_dataset_assert_type(const plot_dataset_type * d, plot_data_type type) {
  if (d->data_type != type)
    util_abort("%s: assert failed - wrong plot type \n",__func__);
}



static int  __make_data_mask(plot_data_type data_type) {
  int mask = 0;
  switch (data_type) {
  case(PLOT_XY):
    mask = PLOT_DATA_X + PLOT_DATA_Y;
    break;
  case(PLOT_XY1Y2):
    mask = PLOT_DATA_X + PLOT_DATA_Y1 + PLOT_DATA_Y2;
    break;
  case(PLOT_X1X2Y):
    mask = PLOT_DATA_X1 + PLOT_DATA_X2 + PLOT_DATA_Y;
    break;
  case(PLOT_XLINE):
    mask = PLOT_DATA_X;
    break;
  case(PLOT_YLINE):
    mask = PLOT_DATA_Y;
    break;
  case(PLOT_HIST):
    mask = PLOT_DATA_X;
    break;
  default:
    util_abort("%s: unrecognized value: %d \n",__func__ , data_type);
  }
  return mask;
}



void plot_dataset_fprintf(const plot_dataset_type * dataset , FILE * stream) {
  fprintf(stream , "    x              y            x1            x2            y1           y2 \n");
  fprintf(stream , "----------------------------------------------------------------------------\n");
  for (int i = 0; i < plot_dataset_get_size(dataset); i++) {
    fprintf(stream , "%12.7f  %12.7f  %12.7f  %12.7f  %12.7f  %12.7f\n",
	    double_vector_safe_iget(dataset->x , i ),
	    double_vector_safe_iget(dataset->y , i ),
	    double_vector_safe_iget(dataset->x1 , i),
	    double_vector_safe_iget(dataset->x2 , i),
	    double_vector_safe_iget(dataset->y1 , i ),
	    double_vector_safe_iget(dataset->y2 , i));
  }
  fprintf(stream , "----------------------------------------------------------------------------\n");
}




/**
 * @return Returns a new plot_dataset_type pointer.
 * @brief Create a new plot_dataset_type
 *
 * Create a new dataset - allocates the memory.
 */
plot_dataset_type *plot_dataset_alloc(plot_data_type data_type , const char* label) {
  plot_dataset_type *d;
  
  d                = util_malloc(sizeof *d , __func__);
  UTIL_TYPE_ID_INIT(d , PLOT_DATASET_TYPE_ID);
  d->data_type     = data_type;
  d->data_mask     = __make_data_mask(data_type);
  d->x   	   = double_vector_alloc( 0 , -1 );
  d->y   	   = double_vector_alloc( 0 , -1 );
  d->x1 	   = double_vector_alloc( 0 , -1 );
  d->x2 	   = double_vector_alloc( 0 , -1 );
  d->y1 	   = double_vector_alloc( 0 , -1 );
  d->y2 	   = double_vector_alloc( 0 , -1 );
  d->label         = util_alloc_string_copy( label );
  /******************************************************************/
  /* Defaults                                                       */
  d->style       = LINE;
  {
    line_attribute_type  line_attr  = {.line_color  = PLOT_DEFAULT_LINE_COLOR  , .line_style  = PLOT_DEFAULT_LINE_STYLE , .line_width  = 1.0};  
    point_attribute_type point_attr = {.point_color = PLOT_DEFAULT_POINT_COLOR , .symbol_type = PLOT_DEFAULT_SYMBOL     , .symbol_size = 1.0};
    
    d->line_attr   = line_attr;
    d->point_attr  = point_attr;
  }
  
  return d;
}

UTIL_SAFE_CAST_FUNCTION(plot_dataset , PLOT_DATASET_TYPE_ID)


/**
   This function frees all the memory related to a dataset - normally
   called automatically from plot_free().
*/
void plot_dataset_free(plot_dataset_type * d)
{
  double_vector_free(d->x);
  double_vector_free(d->x1);
  double_vector_free(d->x2);
  double_vector_free(d->y);
  double_vector_free(d->y1);
  double_vector_free(d->y2);
  free(d->label);
  free(d);
}

void plot_dataset_free__(void * d) {
  plot_dataset_free( plot_dataset_safe_cast( d ));
}


/*****************************************************************/
/** Here comes functions for setting the data for the dataset. */




static void __append_vector(double_vector_type * target, const double * src , const bool_vector_type * mask) {
  if (src == NULL)
    util_abort("%s: trying to extract data from NULL pointer\n",__func__);
  
  
  for (int index = 0; index < bool_vector_size( mask ); index++) 
    if (bool_vector_iget(mask , index)) 
      double_vector_append( target , src[index] );
}



/**
   Before a tuple, i.e. (x,y), (x,y1,y2) , (x1,x2,y), ... is added to
   the dataset we verify that the test isfinite(·) returns true for
   all the elements in the tuple.  

   This functionality is implemented with the boolean vector 'mask'.
*/

void __update_mask( bool_vector_type * mask , const double * data) {
  if (data != NULL) {
    for (int index = 0; index < bool_vector_size( mask ); index++)
      if (!isfinite(data[index])) 
	/* This datapoint is marked as invalid - and not added to the proper datavectors. */
	bool_vector_iset( mask , index , false );  
  }
}


static void plot_dataset_append_vector__(plot_dataset_type * d , int size , const double * x , const double * y , const double * y1 , const double * y2 , const double *x1 , const double *x2) {
  bool_vector_type * mask = bool_vector_alloc( size , true );   /* Initialize to all true */
  
  __update_mask(mask , x);
  __update_mask(mask , y);
  __update_mask(mask , y1);
  __update_mask(mask , y2);
  __update_mask(mask , x1);
  __update_mask(mask , x2);
    
  if (d->data_mask & PLOT_DATA_X)  __append_vector(d->x  , x  ,  mask);
  if (d->data_mask & PLOT_DATA_X1) __append_vector(d->x1 , x1 ,  mask);
  if (d->data_mask & PLOT_DATA_X2) __append_vector(d->x2 , x2 ,  mask);
  if (d->data_mask & PLOT_DATA_Y)  __append_vector(d->y  , y  ,  mask);
  if (d->data_mask & PLOT_DATA_Y1) __append_vector(d->y1 , y1 ,  mask);
  if (d->data_mask & PLOT_DATA_Y2) __append_vector(d->y2 , y2 ,  mask);

  bool_vector_free(mask);
}



/*****************************************************************/
/* Here comes the exported functions - they all have _xy, _xy1y2,
   _xline ... suffix. */


void plot_dataset_append_vector_xy(plot_dataset_type *d , int size, const double * x , const double *y) {
  plot_dataset_assert_type(d , PLOT_XY);
  plot_dataset_append_vector__(d , size , x , y , NULL , NULL , NULL , NULL);
}


void plot_dataset_append_point_xy(plot_dataset_type *d , double x , double y) {
  plot_dataset_append_vector_xy(d , 1 , &x , &y);
}


/*----*/


void plot_dataset_append_vector_xy1y2(plot_dataset_type *d , int size, const double * x , const double *y1 , const double *y2) {
  plot_dataset_assert_type(d , PLOT_XY1Y2);
  plot_dataset_append_vector__(d , size , x , NULL , y1 , y2 , NULL , NULL);
}


void plot_dataset_append_point_xy1y2(plot_dataset_type *d , double x , double y1 , double y2) {
  plot_dataset_append_vector_xy1y2(d , 1 , &x , &y1, &y2);
}



/*----*/

void plot_dataset_append_vector_x1x2y(plot_dataset_type *d , int size, const double * x1 , const double *x2 , const double *y) {
  plot_dataset_assert_type(d , PLOT_X1X2Y);
  plot_dataset_append_vector__(d , size , NULL , y , NULL , NULL , x1 , x2);
}


void plot_dataset_append_point_x1x2y(plot_dataset_type *d , double x1 , double x2 , double y) {
  plot_dataset_append_vector_x1x2y(d , 1 , &x1 , &x2 , &y);
}


/*----*/

void plot_dataset_append_vector_xline(plot_dataset_type *d , int size, const double * x) {
  plot_dataset_assert_type(d , PLOT_XLINE);
  plot_dataset_append_vector__(d , size , x , NULL , NULL , NULL , NULL , NULL);
}


void plot_dataset_append_point_xline(plot_dataset_type *d , double x) {
  plot_dataset_append_vector_xline(d , 1 , &x);
}




/*----*/

void plot_dataset_append_vector_yline(plot_dataset_type *d , int size, const double * y) {
  plot_dataset_assert_type(d , PLOT_YLINE);
  plot_dataset_append_vector__(d , size , NULL , y , NULL , NULL , NULL , NULL );
}


void plot_dataset_append_point_yline(plot_dataset_type *d , double y) {
  plot_dataset_append_vector_yline(d , 1 , &y);
}



/*----*/

/*----*/

void plot_dataset_append_vector_hist(plot_dataset_type *d , int size, const double * x) {
  plot_dataset_assert_type(d , PLOT_HIST);
  plot_dataset_append_vector__(d , size , x , NULL , NULL , NULL , NULL , NULL );
}


void plot_dataset_append_point_hist(plot_dataset_type *d , double x) {
  plot_dataset_append_vector_hist(d , 1 , &x);
}


/*****************************************************************/





void plot_dataset_draw(int stream, plot_dataset_type * d , const plot_range_type * range) {
  const int size = plot_dataset_get_size( d );
  plsstrm(stream);
  
  pllsty(d->line_attr.line_style);                                   /* Setting solid/dashed/... */
  plwid(d->line_attr.line_width * PLOT_DEFAULT_LINE_WIDTH);          /* Setting line width.*/
  plcol0(d->line_attr.line_color);                                   /* Setting line color. */
  plssym(0 , d->point_attr.symbol_size * PLOT_DEFAULT_SYMBOL_SIZE);  /* Setting the size for the symbols. */
  
  
  switch (d->data_type) {
  case(PLOT_XY):
    {
      plot_style_type style       = d->style;
      plot_color_type point_color = d->point_attr.point_color;

      /* 
	 Special case: 
	 -------------
         If only one single point AND plot_style == LINE, we
	 effectively change the plot_style to POINTS (and use the
	 line_color) - otherwise the single point will not be visible.
	 
      */
      
      if ((style == LINE) && (size == 1)) {
	style       = POINTS;
	point_color = d->line_attr.line_color;
	/* The point style will remain at default value. */
      }
      
      /** Starting with the line */    
      if (style == LINE || style == LINE_POINTS) 
	plline(size , double_vector_get_ptr(d->x) , double_vector_get_ptr(d->y));
      /** Then the points. */
      if (style == POINTS || style == LINE_POINTS) {
	plcol0(point_color);       /* Setting the color */
	plpoin(size , double_vector_get_ptr(d->x) , double_vector_get_ptr(d->y) , d->point_attr.symbol_type);
      }
    }
    break;
  case(PLOT_HIST):
    {
      int    bins = (int) sqrt( size );
      double xmin = plot_range_get_xmin( range );
      double xmax = plot_range_get_xmax( range );
      {
        /*
          Could for some fuxxxing reason not get the plhist() function
          to work, and had to resort to the low level plbin function.
        */
        double * limits = util_malloc(sizeof * limits * (bins + 1) , __func__);
        double * x      = util_malloc(sizeof * x * bins , __func__); 
        double * y      = util_malloc(sizeof * y * bins , __func__);
        int i;
        double delta = (xmax - xmin) / bins;
        
        for (i= 0; i <= bins; i++)
          limits[i] = xmin + i*delta;
        
        for (i=0; i < bins; i++) {
          y[i] = 0;
          x[i] = 0.50 * (limits[i] + limits[i + 1]);
        }
        
        
        for (i=0; i < size; i++) {
          double value = double_vector_iget(d->x , i);
          int j;
          for (j = 1; j <= bins; j++)
            if (value < limits[j]) {
              y[j-1]++;
              break;
            }
        }
        
        plbin(bins , x , y , PL_BIN_CENTRED + PL_BIN_NOEXPAND);
        free(x);
        free(y);
        free(limits);
      }
      //plhist(size , double_vector_get_ptr(d->x) , xmin , xmax , bins , 0 /* PL_HIST_DEFAULT */);
    }
    break;
  case(PLOT_XY1Y2):
    plerry(size , double_vector_get_ptr(d->x) , double_vector_get_ptr(d->y1) , double_vector_get_ptr(d->y2));
    break;
  case(PLOT_X1X2Y):
    plerrx(size , double_vector_get_ptr(d->x1) , double_vector_get_ptr(d->x2) , double_vector_get_ptr(d->y) );
    break;
  case(PLOT_YLINE):
    {
      double x[2] = {plot_range_get_xmin(range) , plot_range_get_xmax(range)};
      double y[2];
      
      for (int i=0; i < size; i++) {
	y[0] = double_vector_iget(d->y , i);
	y[1] = double_vector_iget(d->y , i);
	plline(2 , x , y);
      }
    }
    break;
  case(PLOT_XLINE):
    {
      double y[2] = {plot_range_get_ymin(range) , plot_range_get_ymax(range)};
      double x[2];
      
      for (int i=0; i < size; i++) {
	x[0] = double_vector_iget(d->x , i);
	x[1] = double_vector_iget(d->x , i);
	plline(2 , x , y);
      }
    }
    break;
  default:
    util_abort("%s: internal error ... \n",__func__);
    break;
  }
}


void plot_dataset_update_range_histogram(plot_dataset_type * d, plot_range_type * range) {
  plot_range_set_auto_xmin(range , double_vector_get_min( d->x ));
  plot_range_set_auto_xmax(range , double_vector_get_max( d->x ));
  plot_range_set_auto_ymin(range , 0 );
  plot_range_set_auto_ymax(range , double_vector_size( d->x ) / 4);  /* Pure heuristics. */
    
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
void plot_dataset_update_range(plot_dataset_type * d, bool * first_pass , plot_range_type * range) {
  const int size = plot_dataset_get_size( d );
  if (size > 0) {
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
    
    if (d->data_mask & PLOT_DATA_X)  	{x1 = double_vector_get_ptr(d->x);  x2 = double_vector_get_ptr(d->x); }
    if (d->data_mask & PLOT_DATA_X1) 	 x1 = double_vector_get_ptr(d->x1);
    if (d->data_mask & PLOT_DATA_X2) 	 x2 = double_vector_get_ptr(d->x2);


    if (d->data_mask & PLOT_DATA_Y)  {y1 = double_vector_get_ptr(d->y) ;  y2 = double_vector_get_ptr(d->y) ; }
    if (d->data_mask & PLOT_DATA_Y1)  y1 = double_vector_get_ptr(d->y1) ;
    if (d->data_mask & PLOT_DATA_Y2)  y2 = double_vector_get_ptr(d->y2) ;

    if (x1 != NULL) {
      if (*first_pass) {
	/* To ensure sensible initialisation */
	tmp_x_min = x1[0];
	tmp_x_max = x2[0];
      }
      
      for (i=0; i < size; i++) {
	if (x1[i] < tmp_x_min)
	  tmp_x_min = x1[i];
	
	if (x2[i] > tmp_x_max)
	  tmp_x_max = x2[i];
      }
    }


    if (y1 != NULL) {
      if (*first_pass) {
	tmp_y_min = y1[0];
	tmp_y_max = y2[0];
      }
      
      for (i=0; i < size; i++) {
	if (y1[i] < tmp_y_min)
	  tmp_y_min = y1[i];
	
	if (y2[i] > tmp_y_max)
	  tmp_y_max = y2[i];
      }
    }
    
    /**
       If the range value has been set manually these functions
       just return without doing anything.
    */
    plot_range_set_auto_xmin(range , tmp_x_min);
    plot_range_set_auto_xmax(range , tmp_x_max);
    plot_range_set_auto_ymin(range , tmp_y_min);
    plot_range_set_auto_ymax(range , tmp_y_max);
    
    *first_pass = false;
  }
}




