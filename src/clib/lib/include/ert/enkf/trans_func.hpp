#ifndef ERT_TRANS_FUNC_H
#define ERT_TRANS_FUNC_H
#include <stdbool.h>
#include <stdio.h>

#include <ert/util/double_vector.h>

#include <ert/enkf/enkf_types.hpp>

typedef struct trans_func_struct trans_func_type;
typedef double(transform_ftype)(double, const double_vector_type *);
typedef bool(validate_ftype)(const trans_func_type *);

trans_func_type *trans_func_alloc(const stringlist_type *args);
double trans_func_eval(const trans_func_type *trans_func, double x);

void trans_func_free(trans_func_type *trans_func);
bool trans_func_use_log_scale(const trans_func_type *trans_func);
stringlist_type *trans_func_get_param_names(const trans_func_type *trans_func);
double_vector_type *trans_func_get_params(const trans_func_type *trans_func);
const char *trans_func_get_name(const trans_func_type *trans_func);

#endif
