#ifndef __PLOT_CONST_H__
#define __PLOT_CONST_H__

#ifdef __cplusplus 
extern "C" {
#endif




typedef enum {
  plot_xy    = 1,     /* Normal xy plot. */
  plot_xy1y2 = 2,     /* x and y-error bars, from y1 - y2. */
  plot_x1x2y = 3,     /* x error bars (from x1 to x2) and y. */
  plot_xline = 4,     /* Vertical lines with fixed x */
  plot_yline = 5,     /* Horizontal lines with fixed y. */
  plot_hist  = 6      /* A list of values - which are binned, and plotted in a histogram. */    
} plot_data_type;



/**
 * @brief: Plot style for one single graph/dataset.
 * 
 * When adding a new dataset to your plot item you can define what plot style
 * you want for that one graph.
 */
typedef enum plot_style_enum {
    BLANK       = 0,
    LINE        = 1,
    POINTS      = 2,  
    LINE_POINTS = 3
} plot_style_type;


/**
 * @brief: Plot color for one single graph/dataset.
 * 
 * When adding a new dataset to your plot item you can define what plot color
 * you want for that one graph. This color is also used to define color on labels
 * and titles.
 */
typedef enum plot_color_enum {
    WHITE      	= 0,
    RED        	= 1,
    YELLOW     	= 2,
    GREEN      	= 3,
    AQUAMARINE 	= 4,
    PINK    	= 5,
    WHEAT  	= 6,
    GRAY   	= 7,
    BROWN  	= 8,
    BLUE   	= 9,
    VIOLET 	= 10,
    CYAN   	= 11,
    TURQUOISE   = 12,
    MAGENTA 	= 13,
    SALMON  	= 14,
    BLACK   	= 15
} plot_color_type;
#define PLOT_NUM_COLORS 16   /* The number of colors - used for alternating colors. */


/**
   pllsty uses predefined line styles.
   plsty  defines linestyle with pen up/down.
*/

typedef enum  {
  PLOT_LINESTYLE_SOLID_LINE  = 1,
  PLOT_LINESTYLE_SHORT_DASH  = 2,
  PLOT_LINESTYLE_LONG_DASH   = 3
} plot_line_style_type;
     

/* The set of symbols is seemingly extremely limited. */
typedef enum {
  PLOT_SYMBOL_X     	     = 5,
  PLOT_SYMBOL_HDASH 	     = 45,
  PLOT_SYMBOL_FILLED_CIRCLE  = 17 } plot_symbol_type;
  



/* Here comes defaults which apply to the plot as a whole */
#define PLOT_DEFAULT_LABEL_FONTSIZE  0.60    /* Scaled */
#define PLOT_DEFAULT_BOX_COLOR       BLACK
#define PLOT_DEFAULT_LABEL_COLOR     BLACK
#define PLOT_DEFAULT_WIDTH           1024
#define PLOT_DEFAULT_HEIGHT           768


/* Here comes defaults which apply to one dataset. */
#define PLOT_DEFAULT_SYMBOL_SIZE     1.10   /* Scaled */
#define PLOT_DEFAULT_LINE_WIDTH      1.50   /* Scaled */
#define PLOT_DEFAULT_SYMBOL            17   /* PLOT_SYMBOL_FILLED_CIRCLE */
#define PLOT_DEFAULT_LINE_COLOR        BLUE
#define PLOT_DEFAULT_POINT_COLOR       BLUE
#define PLOT_DEAFULT_STYLE             LINE

/* For the variables marked with 'scaled', the API is based on scale
   factors. I.e. to double the symbol size the user would call

     plot_dataset_set_symbol_size(2.0);
     
   Which will multiplu PLOT_DEFAULT_SYMBOL_SIZE with 2.0
*/ 


#ifdef __cplusplus 
}
#endif
#endif
