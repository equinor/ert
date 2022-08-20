#ifndef ERT_RMS_UTIL_H
#define ERT_RMS_UTIL_H

#include <stdio.h>
#include <stdlib.h>

#include <ert/rms/rms_type.hpp>

#define RMS_INACTIVE_DOUBLE -999.00
#define RMS_INACTIVE_FLOAT -999.00
#define RMS_INACTIVE_INT -999

int rms_util_global_index_from_eclipse_ijk(int, int, int, int, int, int);
void rms_util_translate_undef(void *, int, int, const void *, const void *);
void rms_util_fskip_string(FILE *);
int rms_util_fread_strlen(FILE *);
bool rms_util_fread_string(char *, int, FILE *);
void rms_util_fwrite_string(const char *string, FILE *stream);
void rms_util_fwrite_comment(const char *, FILE *);
void rms_util_fwrite_newline(FILE *stream);

#endif
