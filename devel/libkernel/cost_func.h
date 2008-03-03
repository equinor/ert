#ifndef _COST_FUNC_H
#define _COST_FUNC_H
#include "kernel_core.h"
typedef double(f_func_type)(const double);
typedef double(df_func_type)(const  double);

typedef double(reg_func_type)(const int, const double, const double*, const double*);
typedef void(reg_func_gradx_type)(const int,const double,const double*,const double*,double*);

typedef struct cost_func_struct cost_func_type;

typedef enum f_func_enum f_func;
typedef enum reg_func_enum reg_func;

struct cost_func_struct
{
  kernel_list_type *kernel_list;
  f_func_type *f;
  df_func_type *df;
  reg_func_type *reg;
  reg_func_gradx_type *gradx_reg;
};

enum f_func_enum{X,SQRTX,XPLUSSQRTX};
enum reg_func_enum{NONE};

cost_func_type *cost_func_alloc(kernel_list_type*, f_func, reg_func);
void cost_func_free(cost_func_type*);

double cost_func_apply(const cost_func_type*,const double,const int,const double,const double*,const double*);
void cost_func_apply_gradx(const cost_func_type*,const double,const int,const double,const double*,const double*,const int,const double *,const double**,double *g);

double f_func_x(const double);
double f_func_dx(const double);

double f_func_sqrtx(const double);
double f_func_dsqrtx(const double);

double f_func_xplussqrtx(const double);
double f_func_dxplussqrtx(const double);

double reg_func_none(const int,const double,const double*,const double*);
void   reg_func_none_gradx(const int,const double,const double*,const double*,double*);
#endif
