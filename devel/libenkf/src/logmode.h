#ifndef __LOGMODE_H__
#define __LOGMODE_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <enkf_types.h>

typedef struct logmode_struct logmode_type;


logmode_type * logmode_alloc(double , enkf_logmode_enum);
void           logmode_free(logmode_type * );
void           logmode_transform_input_data(const logmode_type *  , int , double *);
void           logmode_transform_output_data(const logmode_type *  , int , double *);
void           logmode_transform_input_distribution(const logmode_type * ,  double *, double *);
double         logmode_transform_output_scalar(const logmode_type *  , double );
logmode_type * logmode_fread_alloc(FILE * );
void           logmode_fwrite(const logmode_type * , FILE *);
bool           logmode_logEnKF(const logmode_type * );

#endif




