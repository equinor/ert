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






/* Default STATIC keywords */
#define NUM_STATIC_KW 3
const char *DEFAULT_STATIC_KW[NUM_STATIC_KW] = {
  "INTEHEAD",
  "LOGIHEAD",
  "PRESSURE"
};

#endif
