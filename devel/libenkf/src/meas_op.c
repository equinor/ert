#include <stdlib.h>
#include <stdio.h>
#include <meas_op.h>

struct meas_op_struct {
  int     size;
  int    *index_list;
  double *weight_list;
}; 


meas_op_type * meas_op_alloc(int size) {
  meas_op_type * meas_op = malloc(sizeof * meas_op);

  meas_op->size = size;
  meas_op->weight_list = malloc(size * sizeof * meas_op->weight_list);
  meas_op->index_list  = malloc(size * sizeof * meas_op->index_list);

  return meas_op;
}


double meas_op_eval(const meas_op_type * meas_op, const double * serial_data) {
  int i;
  double value = 0;
  for (i=0; i < meas_op->size; i++)
    value += serial_data[meas_op->index_list[i]] * meas_op->weight_list[i];
  
  return value;
}


void meas_op_set_scalar(meas_op_type * meas_op, int index) {
  meas_op->index_list[0] = index;
  meas_op->weight_list[0] = 1;
}


void meas_op_free(meas_op_type * meas_op) {
  free(meas_op->index_list);
  free(meas_op->weight_list);
  free(meas_op);
}


