#ifndef RES_UTIL_PRINTF

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {left_pad   = 0,
              right_pad  = 1,
              center_pad = 2} string_alignement_type;

  void util_fprintf_string(const char *  , int , string_alignement_type ,  FILE * );
  void util_fprintf_double(double value , int width , int decimals , char base_fmt , FILE * stream);
  void util_fprintf_int(int value , int width , FILE * stream);

#ifdef __cplusplus
}
#endif



#endif
