#ifndef _KERNEL_CORE_H
#define _KERNEL_CORE_H
typedef double(kernel_func_type)(const int, const double *, const double *, const double);
typedef void(kernel_gradx_type)(const int, const double *, const double *,const double, double *);
typedef void(kernel_gradxx_type)(const int, const double *, const double, double *);

typedef struct kernel_node_struct kernel_node_type;
typedef struct kernel_list_struct kernel_list_type;
typedef enum kernel_type_enum kernel_type;

struct kernel_node_struct
{
  double weight;
  double param;
  kernel_func_type *func;
  kernel_gradx_type *gradx;
  kernel_gradxx_type *gradxx;
};

struct kernel_list_struct
{
  int nk;
  kernel_node_type **kernel_nodes;
};

enum kernel_type_enum{DOT = 0 ,GAUSS = 1};

double  kernel_dotproduct(const int,const double*, const double*, const double);
void    kernel_dotproduct_gradx(const int, const double*, const double*, const double, double *);
void    kernel_dotproduct_gradxx(const int, const double*, const double, double*);

double  kernel_gauss(const int,const double*, const double*, const double);
void    kernel_gauss_gradx(const int, const double*, const double*, const double, double *);
void    kernel_gauss_gradxx(const int, const double*, const double, double*);

double  kernel_tanh(const int,const double*, const double*, const double);
void    kernel_tanh_gradx(const int, const double*, const double*, const double, double *);
void    kernel_tanh_gradxx(const int, const double*, const double, double*);

kernel_node_type * kernel_node_alloc(const kernel_type , double, double);
void               kernel_node_free(kernel_node_type* );
void               kernel_node_assert(const kernel_node_type*);

double  kernel_apply(const kernel_node_type*, const int,const double*, const double*);
void    kernel_apply_gradx(const kernel_node_type*, const int, const double*, const double*,double*);
void    kernel_apply_gradxx(const kernel_node_type*, const int,const double*,double *);

kernel_list_type *  kernel_list_alloc(const int,kernel_type*, double*, double *);
void                kernel_list_free(kernel_list_type*);
void                kernel_list_assert(const kernel_list_type*);

double  kernel_list_apply(const kernel_list_type*, const int, const double*, const double*);
void    kernel_list_apply_gradx(const kernel_list_type*, const int, const double*, const double*,double*);
void    kernel_list_apply_gradxx(const kernel_list_type*, const int, const double*, double*);

double kernel_featurespace_dist_squared(const kernel_list_type*,const int, const double*, const int,const double*,const double**,double*);
double kernel_featurespace_dist_get_aka(const kernel_list_type*,const int, const int, const double*, const double**);
#endif
