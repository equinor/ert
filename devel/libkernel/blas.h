#ifndef _BLAS_H
#define _BLAS_H
// Interface for some of the routines from libblas

extern void dscal_(const int*,const double*,double*,const int*);
extern void daxpy_(const int*,const double*,const double*,const int*,const double*,const int*);
extern void dcopy_(const int*,const double*,const int*,const double*,const int*);
extern double ddot_(const int*,const double*,const int*,const double*,const int*);
#endif

