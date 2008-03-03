#ifndef _PRE_IMAGE_H
#define _PRE_IMAGE_H
void pre_image_approx(const cost_func_type*,const int,const int,const int,const double**,const double*,const double,const double*,const double*,const double*,const int*,double*);

void fw_pre_image_approx_2xdot_xpsqx_(const int*,const int*,const int*,const double*,const double*,const double*,const double*,const int*, double*);
void fw_pre_image_approx_dot_xpsqx_(const int*,const int*,const int*,const double*,const double*,const double*,const double*,const int*, double*);

// Interface to L-BFGS-B routine
extern void setulb_(const int*,     //n
                    const int*,     //m
                    double*,        //x
                    const double*,  //l
                    const double*, //u
                    const int*,     //nbd
                    const double*,  //f
                    const double*,  //g
                    const double*,  //factr
                    const double*,  //pgtol
                    double*,        //wa
                    int*,           //iwa
                    char*,          //task
                    int*,           //iprint
                    char*,          //csave
                    int*,          //lsave
                    int*,           //isave
                    double*         //dsave
                    );
#endif
