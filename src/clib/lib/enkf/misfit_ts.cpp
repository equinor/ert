#include <stdlib.h>

#include <ert/util/double_vector.h>
#include <ert/util/int_vector.h>
#include <ert/util/util.h>

#include <ert/enkf/misfit_ts.hpp>

/** misfit for one ensemble member / observation key. */
struct misfit_ts_struct {
    /** A double vector of length 'history_length' with actual misfit values. */
    double_vector_type *data;
};

misfit_ts_type *misfit_ts_alloc(int history_length) {
    misfit_ts_type *misfit_ts =
        (misfit_ts_type *)util_malloc(sizeof *misfit_ts);

    if (history_length > 0)
        misfit_ts->data = double_vector_alloc(history_length + 1, 0);
    else
        misfit_ts->data =
            NULL; /* Used by the xxx_fread_alloc() function below. */

    return misfit_ts;
}

misfit_ts_type *misfit_ts_fread_alloc(FILE *stream) {
    misfit_ts_type *misfit_ts = misfit_ts_alloc(0);
    if (misfit_ts->data == NULL)
        misfit_ts->data = double_vector_fread_alloc(stream);
    return misfit_ts;
}

void misfit_ts_fwrite(const misfit_ts_type *misfit_ts, FILE *stream) {
    double_vector_fwrite(misfit_ts->data, stream);
}

static void misfit_ts_free(misfit_ts_type *misfit_ts) {
    double_vector_free(misfit_ts->data);
    free(misfit_ts);
}

void misfit_ts_free__(void *vector) {
    misfit_ts_free(static_cast<misfit_ts_type *>(vector));
}

void misfit_ts_iset(misfit_ts_type *vector, int time_index, double value) {
    double_vector_iset(vector->data, time_index, value);
}

/* Step2 is inclusive */
double misfit_ts_eval(const misfit_ts_type *vector,
                      const int_vector_type *steps) {
    double misfit_sum = 0;
    int step;

    for (int i = 0; i < int_vector_size(steps); ++i) {
        step = int_vector_iget(steps, i);
        misfit_sum += double_vector_iget(vector->data, step);
    }

    return misfit_sum;
}
