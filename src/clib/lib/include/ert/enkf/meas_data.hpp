#ifndef ERT_MEAS_DATA_H
#define ERT_MEAS_DATA_H

#include <Eigen/Dense>
#include <stdbool.h>
#include <vector>

#include <ert/util/bool_vector.h>
#include <ert/util/hash.h>

#include <ert/tooling.hpp>

typedef struct meas_data_struct meas_data_type;
typedef struct meas_block_struct meas_block_type;

meas_block_type *meas_block_alloc(const char *obs_key,
                                  const std::vector<size_t> &realizations,
                                  int obs_size);
void meas_block_iset(meas_block_type *meas_block, int iens, int iobs,
                     double value);
double meas_block_iget(const meas_block_type *meas_block, int iens, int iobs);
double meas_block_iget_ens_mean(meas_block_type *meas_block, int iobs);
double meas_block_iget_ens_std(meas_block_type *meas_block, int iobs);
void meas_block_deactivate(meas_block_type *meas_block, int iobs);
bool meas_block_iget_active(const meas_block_type *meas_block, int iobs);
void meas_block_free(meas_block_type *meas_block);

meas_data_type *meas_data_alloc(const std::vector<size_t> &realiations);

void meas_data_free(meas_data_type *);
Eigen::MatrixXd meas_data_makeS(const meas_data_type *matrix);
meas_block_type *meas_data_add_block(meas_data_type *matrix,
                                     const char *obs_key, int report_step,
                                     int obs_size);
meas_block_type *meas_data_iget_block(const meas_data_type *matrix,
                                      int block_mnr);

int meas_block_get_total_obs_size(const meas_block_type *meas_block);
#endif
