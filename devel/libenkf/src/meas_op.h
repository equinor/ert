#ifndef __MEAS_OP_H__
#define __MEAS_OP_H__

typedef struct meas_op_struct meas_op_type;

meas_op_type * meas_op_alloc(int);
void           meas_op_free(meas_op_type *);
double         meas_op_eval(const meas_op_type * , const double *);
void           meas_op_set_scalar(meas_op_type * , int );

#endif

