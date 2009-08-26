/**
   This file contains verious default values which are compiled into
   the enkf executable. Everytime you add something here you should
   ask yourself:

    * Can we determine this automagically.
    * Should the user be required to enter this information.
    
*/

#ifndef __ENKF_DEFAULT__
#define __ENKF_DEFAULT__
#include <stdbool.h>

#define DEFAULT_HOST_TYPE  "STATOIL"
#define DEFAULT_DBASE_TYPE "PLAIN"



/* Eclipse IO  related stuff */
#define DEFAULT_FORMATTED   false
#define DEFAULT_ENDIAN_FLIP true
#define DEFAULT_UNIFIED     false



/* Where the history is coming from - default value for config item: HISTORY_SOURCE */
#define DEFAULT_HISTORY_SOURCE "SCHEDULE"



/* The number of times to resubmit - default value for config item: MAX_SUBMIT */
#define DEFAULT_MAX_SUBMIT "2"


#define DEFAULT_RESAMPLE_WHEN_FAIL  "FALSE"
#define DEFAULT_MAX_INTERNAL_SUBMIT "1"      /** Attached to keyword : MAX_RETRY */


#define DEFAULT_LOG_LEVEL "1"



/* Defaults for the EnKF analysis. */
#define DEFAULT_ENKF_MODE          "STANDARD"
#define DEFAULT_ENKF_TRUNCATION    "0.99"    /* NB String */
#define DEFAULT_ENKF_ALPHA         "1.50"    /* NB String */
#define DEFAULT_MERGE_OBSERVATIONS "False"   /* NB - string representation of TRUE|FALSE */
#define DEFAULT_RERUN              "False"
#define DEFAULT_RERUN_START        "0"   


/* Default directories. */
#define DEFAULT_PLOT_PATH   "plots"
#define DEFAULT_RUNPATH     "simulations/realization%d"
#define DEFAULT_ENSPATH     "storage"


#define DEFAULT_PLOT_WIDTH   1024
#define DEFAULT_PLOT_HEIGHT   768
#define DEFAULT_IMAGE_TYPE   "png"
#define DEFAULT_PLOT_DRIVER  "PLPLOT"


/* The magic string used to signal that *ALL* static keywords should be included. */
#define DEFAULT_ALL_STATIC_KW "__ALL__"
#define NUM_STATIC_KW          56


/* 
   The whole thing is defined as one literal - just because I don't
   understand C-linkage.
*/
#define DEFAULT_STATIC_KW (const char *[NUM_STATIC_KW]) { \
  "ACAQ",   	\
  "ACAQNUM",    \
  "DOUBHEAD",   \
  "ENDSOL",     \
  "HIDDEN",     \
  "IAAQ",   	\
  "ICAQ",   	\
  "ICAQNUM",    \
  "ICON",   	\
  "IGRP",   	\
  "ILBR",   	\
  "ILBS",   	\
  "INTEHEAD",   \
  "ISEG",   	\
  "ISTHG",      \
  "ISTHW",   	\
  "IWEL",   	\
  "LOGIHEAD",   \
  "PRESSURE",   \
  "RS",   	\
  "RSEG",   	\
  "RV",   	\
  "SAAQ",   	\
  "SCAQ",   	\
  "SCAQNUM",    \
  "SCON",   	\
  "SGAS",   	\
  "SGRP",   	\
  "STARTSOL",   \
  "SWAT",   	\
  "SWEL",   	\
  "XAAQ",   	\
  "XCON",   	\
  "XGRP",   	\
  "XWEL",   	\
  "ZGRP",   	\
  "ZWEL",       \
  "ENDLGR",     \
  "LGR",        \
  "LGRHEADD",   \
  "LGRHEADI",   \
  "LGRHEADQ",   \
  "LGRNAMES",   \
  "EOWC"    ,   \
  "IOWC"    ,   \
  "OWC"     ,   \
  "ZNODE"   ,   \
  "INODE"   ,   \
  "RNODE"   ,   \
  "LGWEL"   ,   \
  "IBRAN"   ,   \
  "INOBR"   ,   \
  "RBRAN"   ,   \
  "PRESROCC",   \
  "REGDIMS" ,   \
  "THRESHPR"}



/* 
   The string added at the beginning and end of string which should be
   replaced with the template parser.  
*/

#define DEFAULT_START_TAG "<"
#define DEFAULT_END_TAG ">"


/*****************************************************************/
/* file system spesific defaults. */

/**
  Name of the default case.
*/

#define DEFAULT_CASE "default"

#define DEFAULT_PLAIN_PARAMETER_PATH 	      "%04d/mem%03d/Parameter"
#define DEFAULT_PLAIN_STATIC_PATH    	      "%04d/mem%03d/Static"
#define DEFAULT_PLAIN_DYNAMIC_FORECAST_PATH "%04d/mem%03d/Forecast"
#define DEFAULT_PLAIN_DYNAMIC_ANALYZED_PATH "%04d/mem%03d/Analyzed"
#define DEFAULT_PLAIN_INDEX_PATH            "%04d/mem%03d/INDEX"


#define DEFAULT_SQLITE_PARAMETER_DBFILE  	 "Parameter.sqlite_db"
#define DEFAULT_SQLITE_STATIC_DBFILE     	 "Static.sqlite_db"
#define DEFAULT_SQLITE_DYNAMIC_FORECAST_DBFILE   "Forecast.sqlite_db"
#define DEFAULT_SQLITE_DYNAMIC_ANALYZED_DBFILE   "Analyzed.sqlite_db"



#endif
