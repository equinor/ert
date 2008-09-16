
#ifndef __HISTORY_ENS_DIAG_H__
#define __HISTORY_ENS_DIAG_H__

#include <stdbool.h>
#include <ecl_sum.h>
#include <history.h>

/*void history_ens_diag_ens(int  , int , const char *  , int , const char **, int , const char **, const char *, const char *, const char * ,  bool , bool , bool , bool);*/
/*void history_ens_diag_ens(int , int  , const char ** , const char ** , const char * , const char * , const char * , bool , bool , bool , bool);*/
void history_ens_diag_ens_interactive(const char* , const char * , bool , bool , bool, history_type * );
void history_ens_diag_avg_production_interactive(const char * , const char * , const char * , const char * , bool , bool );
ecl_sum_type ** history_ens_diag_load_ensemble(int iens1, int iens2 , int * _min_size , const char *ens_path , const char *eclbase_dir , const char *eclbase , bool report_mode ,  bool fmt_file, bool unified , bool endian_convert, history_type * );
void history_ens_diag_imake_plotfile(int iens1 , int iens2 , int min_size , const ecl_sum_type **ecl_sum_list , const char *out_file , const char * title , const char * well , const char * var, int index, history_type * );
ecl_sum_type ** history_ens_diag_load_ensemble2(int , int , int  , int * , const char * , const char *, const char *, bool ,  bool , bool , bool , history_type * );

/*
  void history_ens_diag_make_gnuplot_interactive();
*/

#endif
