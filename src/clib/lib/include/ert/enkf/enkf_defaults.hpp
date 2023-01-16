/**
   This file contains verious default values which are compiled into
   the enkf executable.
*/

#ifndef ERT_ENKF_DEFAULT
#define ERT_ENKF_DEFAULT
#include <stdbool.h>

/**
   The format string used when creating "search-strings" which should
   be replaced in the gen_kw template files - MUST contain one %s
   placeholder which will be replaced with the parameter name.
*/
#define DEFAULT_GEN_KW_TAG_FORMAT "<%s>"

#endif
