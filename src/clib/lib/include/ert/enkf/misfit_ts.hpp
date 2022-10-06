#ifndef ERT_MISFIT_TS_H
#define ERT_MISFIT_TS_H

#include <stdio.h>

typedef struct misfit_ts_struct misfit_ts_type;

void misfit_ts_fwrite(const misfit_ts_type *misfit_ts, FILE *stream);
double misfit_ts_eval(const misfit_ts_type *ts, const int_vector_type *steps);
misfit_ts_type *misfit_ts_alloc(int history_length);
misfit_ts_type *misfit_ts_fread_alloc(FILE *stream);
void misfit_ts_free__(void *vector);
void misfit_ts_iset(misfit_ts_type *vector, int time_index, double value);

#endif
