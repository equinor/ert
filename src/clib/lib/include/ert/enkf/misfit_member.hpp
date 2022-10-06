#ifndef ERT_MISFIT_MEMBER_H
#define ERT_MISFIT_MEMBER_H

#include <stdio.h>

#include <ert/enkf/misfit_ts.hpp>

typedef struct misfit_member_struct misfit_member_type;
misfit_ts_type *misfit_member_get_ts(const misfit_member_type *member,
                                     const char *obs_key);
bool misfit_member_has_ts(const misfit_member_type *member,
                          const char *obs_key);
misfit_member_type *misfit_member_fread_alloc(FILE *stream);
void misfit_member_fwrite(const misfit_member_type *node, FILE *stream);
void misfit_member_update(misfit_member_type *node, const char *obs_key,
                          int history_length, int iens,
                          const double **work_chi2);
void misfit_member_free__(void *node);
misfit_member_type *misfit_member_alloc(int iens);

#endif
