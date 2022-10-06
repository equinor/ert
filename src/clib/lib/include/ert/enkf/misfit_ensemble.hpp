#ifndef ERT_MISFIT_ENSEMBLE_H
#define ERT_MISFIT_ENSEMBLE_H

#include <stdbool.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/misfit_member.hpp>

#include <ert/enkf/misfit_ensemble_typedef.hpp>

void misfit_ensemble_fread(misfit_ensemble_type *misfit_ensemble, FILE *stream);
void misfit_ensemble_clear(misfit_ensemble_type *table);
misfit_ensemble_type *misfit_ensemble_alloc();
void misfit_ensemble_free(misfit_ensemble_type *table);
void misfit_ensemble_fwrite(const misfit_ensemble_type *misfit_ensemble,
                            FILE *stream);
bool misfit_ensemble_initialized(const misfit_ensemble_type *misfit_ensemble);

void misfit_ensemble_initialize(misfit_ensemble_type *misfit_ensemble,
                                const ensemble_config_type *ensemble_config,
                                const enkf_obs_type *enkf_obs, enkf_fs_type *fs,
                                int ens_size, int history_length,
                                bool force_init);

void misfit_ensemble_set_ens_size(misfit_ensemble_type *misfit_ensemble,
                                  int ens_size);
int misfit_ensemble_get_ens_size(const misfit_ensemble_type *misfit_ensemble);

misfit_member_type *
misfit_ensemble_iget_member(const misfit_ensemble_type *table, int iens);

#endif
