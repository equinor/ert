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




/* Eclipse IO  related stuff */
#define DEFAULT_FORMATTED   false
#define DEFAULT_ENDIAN_FLIP true
#define DEFAULT_UNIFIED     false








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


#endif
