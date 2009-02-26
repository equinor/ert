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


/* Set the default forward model to ECLIPSE 100. */
#define DEFAULT_FORWARD_MODEL "ECLIPSE100"




/* Eclipse IO  related stuff */
#define DEFAULT_FORMATTED   false
#define DEFAULT_ENDIAN_FLIP true
#define DEFAULT_UNIFIED     false



/* Where the history is coming from - default value for config item: HISTORY_SOURCE */
#define DEFAULT_HISTORY_SOURCE "SCHEDULE"



/* The number of times to resubmit - default value for config item: MAX_SUBMIT */
#define DEFAULT_MAX_SUBMIT "2"



/* Defaults for the EnKF analysis. */
#define DEFAULT_ENKF_MODE          "STANDARD"
#define DEFAULT_ENKF_TRUNCATION    "0.99"    /* NB String */
#define DEFAULT_ENKF_ALPHA         "1.50"    /* NB String */
#define DEFAULT_MERGE_OBSERVATIONS "False"   /* NB - string representation of TRUE|FALSE */

/* Default directories. */
#define DEFAULT_PLOT_PATH   "plots"
#define DEFAULT_RESULT_PATH "results/step_%d"
#define DEFAULT_RUNPATH     "simulations/realization-%d-step-%d-to-%d"
#define DEFAULT_ENSPATH     "storage"



/* The magic string used to signal that *ALL* static keywords should be included. */
#define DEFAULT_ALL_STATIC_KW "__ALL__"
#define NUM_STATIC_KW         37


/* 
   The whole thing is defined as one literal - just because I don't
   understand C-linkage.
*/
#define DEFAULT_STATIC_KW (const char *[NUM_STATIC_KW]) { \
  "INTEHEAD",   \
  "LOGIHEAD",   \
  "DOUBHEAD",   \
  "IGRP",   	\
  "SGRP",   	\
  "XGRP",   	\
  "ZGRP",   	\
  "IWEL",   	\
  "SWEL",   	\
  "XWEL",   	\
  "ZWEL",   	\
  "ICON",   	\
  "SCON",   	\
  "XCON",   	\
  "HIDDEN",     \
  "STARTSOL",   \
  "PRESSURE",   \
  "SWAT",   	\
  "SGAS",   	\
  "RS",   	\
  "RV",   	\
  "ENDSOL",     \
  "ICAQNUM",    \
  "IAAQ",   	\
  "ICAQ",   	\
  "SCAQNUM",    \
  "SAAQ",   	\
  "SCAQ",   	\
  "ACAQNUM",    \
  "XAAQ",   	\
  "ACAQ",   	\
  "ISEG",   	\
  "ILBS",   	\
  "ILBR",   	\
  "RSEG",   	\
  "ISTHW",   	\
  "ISTHG"}       




/* 
   The string added at the beginning and end of string which should be
   replaced with the template parser.  
*/

#define DEFAULT_START_TAG "<"
#define DEFAULT_END_TAG ">"


/**
  Name of the default case.
*/

#define DEFAULT_CASE "default"





#endif
