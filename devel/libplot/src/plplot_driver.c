#include <stdlib.h>
#include <util.h>
#include <string.h>
#include <math.h>
#include <plplot/plplot.h>
#include <plot_driver.h>
#include <plot_const.h>
#include <arg_pack.h>


typedef struct {
  int          stream;     /* Always zero ?? */
  char       * plbox_xopt;
  char       * plbox_yopt;
  char       * filename;         /* Filename for the plot */    
  char       * device;           /* Type for the plot: jpg|png|xwin|... */
} plplot_state_type;





static plplot_state_type * plplot_state_alloc( void * init_arg ) {
  plplot_state_type * state = util_malloc( sizeof * state , __func__);
  state->stream     = 0;
  state->plbox_xopt = util_alloc_string_copy( PLOT_DEFAULT_PLBOX_XOPT );
  state->plbox_yopt = util_alloc_string_copy( PLOT_DEFAULT_PLBOX_YOPT );
  {
    arg_pack_type * arg_pack = arg_pack_safe_cast( init_arg );
    state->filename          = util_alloc_string_copy( arg_pack_iget_ptr( arg_pack , 0) );
    state->device            = util_alloc_string_copy( arg_pack_iget_ptr( arg_pack , 1) );
    
    plsstrm(state->stream);
    plsdev(state->device);    /* Can this be NULL?? */
    if (strcmp(state->device , "xwin") != 0)
      plsfnam(state->filename);
  }

  /** This color initialization must be here - do not really understand what for. */
  plscol0(WHITE, 255, 255, 255);
  plscol0(BLACK, 0, 0, 0);
  plfontld(0);
  //plinit();
  return state;
}


static void plplot_state_close( plplot_state_type * state ) {
  plsstrm(state->stream);
  plend1();
  plend();
  
  util_safe_free( state->filename );
  util_safe_free( state->device );
  free( state->plbox_xopt );
  free( state->plbox_yopt );
  free( state );
}



/*****************************************************************/




static void plplot_close_driver( plot_driver_type * driver ) {
  plplot_state_close( driver->state );
}



static void plplot_set_window_size( plot_driver_type * driver , int width , int height) {
  char * geometry = util_alloc_sprintf("%dx%d", width, height);
  plsetopt("geometry", geometry);
  free(geometry);

  {
    plplot_state_type * state = driver->state;
    plsstrm(state->stream);  
    plinit();
    pladv(0);  /* And what does this do ... */
    plvsta();
  }
}



static void plplot_set_labels( plot_driver_type * driver , const char * title , const char * xlabel , const char * ylabel , plot_color_type label_color , double label_font_size) {
  plcol0(label_color);
  plschr(0, label_font_size * PLOT_DEFAULT_LABEL_FONTSIZE);
  pllab(xlabel, ylabel, title);
}



static void plplot_set_axis(plot_driver_type * driver , plot_range_type * range , const char * timefmt , plot_color_type box_color , double tick_font_size) {
  plplot_state_type * state = driver->state;
  plwind( plot_range_get_final_xmin( range ) , plot_range_get_final_xmax( range ) , plot_range_get_final_ymin( range ) , plot_range_get_final_ymax( range ));
  
  plcol0(box_color);
  plschr(0, tick_font_size * PLOT_DEFAULT_LABEL_FONTSIZE);
  
  if (timefmt != NULL) {
    pltimefmt(timefmt);
    state->plbox_xopt = util_realloc_sprintf( state->plbox_xopt , "%s%c" , state->plbox_xopt , 'd');
  }
  plbox(state->plbox_xopt, 0.0, 0, state->plbox_yopt , 0.0, 0);
  
}

/*****************************************************************/

static void plplot_setup_linestyle( line_attribute_type line_attr ) {
  pllsty(line_attr.line_style);                                   /* Setting solid/dashed/... */
  plwid(line_attr.line_width * PLOT_DEFAULT_LINE_WIDTH);          /* Setting line width.*/
  plcol0(line_attr.line_color);                                   /* Setting line color. */
}



static void plplot_setup_pointstyle( point_attribute_type point_attr ) {
  plssym(0 , point_attr.symbol_size * PLOT_DEFAULT_SYMBOL_SIZE);  /* Setting the size for the symbols. */
  plcol0(point_attr.point_color);                                 /* Setting color for points. */
}

/*****************************************************************/


void plplot_plot_xy1y2(plot_driver_type * driver     , 
                       const char * label , 
                       const double_vector_type * x  , 
                       const double_vector_type * y1  , 
                       const double_vector_type * y2  , 
                       line_attribute_type line_attr) {

  int size = double_vector_size( x );
  plplot_setup_linestyle( line_attr );
  plerry(size , double_vector_get_ptr(x) , double_vector_get_ptr(y1) , double_vector_get_ptr(y2));
}





void plplot_plot_x1x2y(plot_driver_type * driver      , 
                       const char * label             , 
                       const double_vector_type * x1  , 
                       const double_vector_type * x2  , 
                       const double_vector_type * y   , 
                       line_attribute_type line_attr) {

  int size = double_vector_size( x1 );
  plplot_setup_linestyle( line_attr );
  plerrx(size , double_vector_get_ptr(x1) , double_vector_get_ptr(x2) , double_vector_get_ptr(y));
}






void plplot_plot_xy(plot_driver_type * driver     , 
                    const char * label , 
                    const double_vector_type * x  , 
                    const double_vector_type * y  , 
                    plot_style_type style         , 
                    line_attribute_type line_attr , 
                    point_attribute_type point_attr) {

  int size = double_vector_size( x );


  /* 
     Special case: 
     -------------
     If only one single point AND plot_style == LINE, we
     effectively change the plot_style to POINTS (and use the
     line_color) - otherwise the single point will not be visible.
  */
  
  if ((style == LINE) && (size == 1)) {
    style                  = POINTS;
    point_attr.point_color = line_attr.line_color;
  }
  
  if (style & LINE) {
    plplot_setup_linestyle( line_attr );
    plline(size , double_vector_get_ptr(x) , double_vector_get_ptr(y));
  }

  if (style & POINTS) {
    plplot_setup_pointstyle( point_attr );
    plpoin(size , double_vector_get_ptr(x) , double_vector_get_ptr(y) , point_attr.symbol_type);
  }
}



void plplot_plot_hist( plot_driver_type * driver, const char * label , const double_vector_type * x , line_attribute_type line_attr) {
  int size = double_vector_size( x );
  plplot_setup_linestyle( line_attr );
  {
    int    bins = (int) sqrt( size );
    double xmin = double_vector_get_min( x );
    double xmax = double_vector_get_max( x );
    {
      /*
        Could for some fuxxxing reason not get the plhist() function
          to work, and had to resort to the low level plbin function.
      */
      double * limits  = util_malloc(sizeof * limits * (bins + 1) , __func__);
      double * x_      = util_malloc(sizeof * x_ * bins , __func__); 
      double * y_      = util_malloc(sizeof * y_ * bins , __func__);
      int i;
      double delta = (xmax - xmin) / bins;
      
      for (i= 0; i <= bins; i++)
        limits[i] = xmin + i*delta;
      
      for (i=0; i < bins; i++) {
        y_[i] = 0;
        x_[i] = 0.50 * (limits[i] + limits[i + 1]);
      }
      
      
      for (i=0; i < size; i++) {
        double value = double_vector_iget(x , i);
        int j;
        for (j = 1; j <= bins; j++)
          if (value < limits[j]) {
            y_[j-1]++;
            break;
          }
      }
      
      /*
        for (i = 0; i < bins; i++)
        printf("x[%d] = %g    y[%d] = %g\n",i,x_[i],i,y_[i]);
      */
      
      plbin(bins , x_ , y_ , PL_BIN_CENTRED + PL_BIN_NOEXPAND);
      free(x_);
      free(y_);
      free(limits);
    }
    //plhist(size , double_vector_get_ptr(d->x) , xmin , xmax , bins , 0 /* PL_HIST_DEFAULT */);
  }
}




plot_driver_type * plplot_driver_alloc(void * init_arg) {
  plot_driver_type * driver = plot_driver_alloc_empty(PLPLOT , "PLPLOT");
  driver->state           = plplot_state_alloc( init_arg );
  
  driver->close_driver 	  = plplot_close_driver;
  driver->set_window_size = plplot_set_window_size;
  driver->set_labels      = plplot_set_labels; 
  driver->set_axis        = plplot_set_axis;
  
  driver->plot_xy         = plplot_plot_xy;
  driver->plot_xy1y2      = plplot_plot_xy1y2;
  driver->plot_x1x2y      = plplot_plot_x1x2y;
  driver->plot_hist       = plplot_plot_hist;

  return driver;
}
