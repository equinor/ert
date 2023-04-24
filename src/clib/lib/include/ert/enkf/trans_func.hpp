#ifndef ERT_TRANS_FUNC_H
#define ERT_TRANS_FUNC_H
#include <ert/enkf/enkf_types.hpp>
#include <ert/util/stringlist.h>
#include <stdbool.h>
#include <stdio.h>
#include <vector>

typedef struct trans_func_struct trans_func_type;
typedef double(transform_ftype)(double, const std::vector<double>);
typedef bool(validate_ftype)(const trans_func_type *);

trans_func_type *trans_func_alloc(const stringlist_type *args);
double trans_func_eval(const trans_func_type *trans_func, double x);

void trans_func_free(trans_func_type *trans_func);

#endif
