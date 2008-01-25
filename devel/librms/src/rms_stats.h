#ifndef __RMS_STATS_H__
#define __RMS_STATS_H__
#include <rms_tagkey.h>

void rms_stats_mean_std(rms_tagkey_type * , rms_tagkey_type * , const char * , int , const char ** , bool);
void rms_stats_update_ens(const char * , const char *, const char **, const char *, int , const double **);

#endif
