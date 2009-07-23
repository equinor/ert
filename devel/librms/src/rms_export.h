#ifndef __RMS_EXPORT_H__
#define __RMS_EXPORT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <ecl_grid.h>
#include <ecl_kw.h>

  
void rms_export_roff_from_keyword(const char *filename, ecl_grid_type *ecl_grid, 
    ecl_kw_type **ecl_kw, int size);



#ifdef __cplusplus
}
#endif
#endif

