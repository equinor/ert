#ifndef __MEAS_DATA_H__
#define __MEAS_DATA_H__
#include <stdio.h>

typedef struct meas_vector_struct meas_vector_type;

meas_vector_type * meas_vector_alloc();
void           meas_vector_free(meas_vector_type *);
void           meas_vector_add(meas_vector_type * , double );
void           meas_vector_reset(meas_vector_type * );
void           meas_vector_fprintf(const meas_vector_type *  , FILE *);
int            meas_vector_get_nrobs(const meas_vector_type * );
const double * meas_vector_get_data_ref(const meas_vector_type * );
#endif
