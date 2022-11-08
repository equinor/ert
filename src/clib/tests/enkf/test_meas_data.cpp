#include "catch2/catch.hpp"
#include <numeric>

#include <ert/enkf/meas_data.hpp>

void meas_block_calculate_ens_stats(meas_block_type *meas_block);

TEST_CASE("meas_block_calculate_ens_stats", "[meas_data]") {

    int ens_size = 2;
    std::vector<size_t> realizations(ens_size);
    std::iota(realizations.begin(), realizations.end(), 0);
    int obs_size = 3;
    const char *obs_key = "OBS1";
    auto *mb = meas_block_alloc(obs_key, realizations, obs_size);

    meas_block_iset(mb, 0, 0, 1);
    meas_block_iset(mb, 0, 1, 2);
    meas_block_iset(mb, 0, 2, 3);
    meas_block_iset(mb, 1, 0, 3.5);
    meas_block_iset(mb, 1, 1, 4.5);
    meas_block_iset(mb, 1, 2, 6);

    meas_block_calculate_ens_stats(mb);

    REQUIRE(meas_block_iget_ens_mean(mb, 0) == 2.25);
    REQUIRE(meas_block_iget_ens_std(mb, 0) == 1.25);

    REQUIRE(meas_block_iget_ens_mean(mb, 1) == 3.25);
    REQUIRE(meas_block_iget_ens_std(mb, 1) == 1.25);

    REQUIRE(meas_block_iget_ens_mean(mb, 2) == 4.5);
    REQUIRE(meas_block_iget_ens_std(mb, 2) == 1.5);
}
