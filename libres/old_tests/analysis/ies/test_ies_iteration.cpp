#include <algorithm>
#include <ert/util/rng.h>
#include <ert/util/test_util.hpp>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>
#include <ert/res_util/es_testdata.hpp>

void init_stdA(const res::es_testdata &testdata, Eigen::MatrixXd &A2) {
    ies::Config ies_config(false);
    ies_config.set_truncation(1.00);

    int active_ens_size = A2.cols();
    Eigen::MatrixXd W0 =
        Eigen::MatrixXd::Zero(active_ens_size, active_ens_size);
    Eigen::MatrixXd X =
        ies::makeX(A2, testdata.S, testdata.R, testdata.E, testdata.D,
                   ies_config.inversion, ies_config.get_truncation(), W0, 1, 1);

    A2 *= X;
}

/*
  This function will run the forward model again and update the matrices in the
  testdata structure - with A1 as prior.
*/

void eigen_copy_column(Eigen::MatrixXd &dst_matrix,
                       const Eigen::MatrixXd &src_matrix, int dst_column,
                       int src_column) {
    for (int row = 0; row < dst_matrix.rows(); row++)
        dst_matrix(row, dst_column) = src_matrix(row, src_column);
}

void forward_model(res::es_testdata &testdata, const Eigen::MatrixXd &A1) {
    int nrens = A1.cols();
    int ndim = A1.rows();
    int nrobs = testdata.S.rows();

    /* Model prediction gives new S given prior S=func(A) */
    for (int iens = 0; iens < nrens; iens++) {
        for (int i = 0; i < nrobs; i++) {
            double coeffa = A1(0, iens);
            double coeffb = A1(1, iens);
            double coeffc = A1(2, iens);
            double y = coeffa * i * i + coeffb * i + coeffc;
            testdata.S(i, iens) = y;
        }
    }

    /* Updating D according to new S: D=dObs+E-S*/
    for (int i = 0; i < nrens; i++)
        eigen_copy_column(testdata.D, testdata.dObs, i, 0);

    testdata.D += testdata.E;
    testdata.D -= testdata.S;
}

void cmp_std_ies(res::es_testdata &testdata) {

    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    Eigen::MatrixXd A1 = testdata.make_state("prior");
    Eigen::MatrixXd A2 = testdata.make_state("prior");

    ies::data::Data ies_data(testdata.active_ens_size);
    ies::Config ies_config(true);

    forward_model(testdata, A1);
    ies_config.set_truncation(1.0);
    ies_config.max_steplength = 0.6;
    ies_config.min_steplength = 0.6;
    ies_config.inversion = ies::IES_INVERSION_EXACT;

    /* ES solution */
    int num_iter = 100;
    init_stdA(testdata, A2);

    for (int iter = 0; iter < num_iter; iter++) {
        forward_model(testdata, A1);
        ies::init_update(ies_data, testdata.ens_mask, testdata.obs_mask);
        ies::updateA(ies_data, A1, testdata.S, testdata.R, testdata.E,
                     testdata.D, ies_config.inversion,
                     ies_config.get_truncation(),
                     ies_config.get_steplength(ies_data.iteration_nr));
        ies_data.iteration_nr++;

        if (A1.isApprox(A2, 1e-5))
            break;
    }

    test_assert_true(A1.isApprox(A2, 1e-5));

    rng_free(rng);
}

void cmp_std_ies_delrel(res::es_testdata &testdata) {
    int num_iter = 100;
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    Eigen::MatrixXd A1 = testdata.make_state("prior");
    Eigen::MatrixXd A2 = testdata.make_state("prior");
    Eigen::MatrixXd A1c = A1;
    Eigen::MatrixXd A2c = A2;
    ies::data::Data ies_data(testdata.active_ens_size);
    ies::Config ies_config(true);

    forward_model(testdata, A1);
    ies_config.set_truncation(1.0);
    ies_config.min_steplength = 0.6;
    ies_config.max_steplength = 0.6;
    ies_config.inversion = ies::IES_INVERSION_EXACT;
    int iens_deact = testdata.active_ens_size / 2;

    /* IES solution after with one realization is inactivated */
    for (int iter = 0; iter < num_iter; iter++) {
        forward_model(testdata, A1);

        // Removing the realization
        if (iter == 6) {
            testdata.deactivate_realization(iens_deact);
            A1c = Eigen::MatrixXd::Zero(
                A1.rows(), std::count(testdata.ens_mask.begin(),
                                      testdata.ens_mask.end(), true));
            int iens_active = 0;
            for (int iens = 0; iens < A1.cols(); iens++) {
                if (testdata.ens_mask[iens]) {
                    eigen_copy_column(A1c, A1, iens_active, iens);
                    iens_active += 1;
                }
            }
            A1 = A1c;
        }

        ies::init_update(ies_data, testdata.ens_mask, testdata.obs_mask);
        ies::updateA(ies_data, A1, testdata.S, testdata.R, testdata.E,
                     testdata.D, ies_config.inversion,
                     ies_config.get_truncation(),
                     ies_config.get_steplength(ies_data.iteration_nr));
        ies_data.iteration_nr++;
    }

    /* ES update with one realization removed*/
    {
        A2c = Eigen::MatrixXd::Zero(A2.rows(),
                                    std::count(testdata.ens_mask.begin(),
                                               testdata.ens_mask.end(), true));
        int iens_active = 0;
        for (int iens = 0; iens < A2.cols(); iens++) {
            if (testdata.ens_mask[iens]) {
                eigen_copy_column(A2c, A2, iens_active, iens);
                iens_active += 1;
            }
        }
        A2 = A2c;
    }
    forward_model(testdata, A2);

    init_stdA(testdata, A2);

    test_assert_true(A1.isApprox(A2, 1e-5));
    rng_free(rng);
}

/*
  This test verifies that the update iteration do not crash hard when
  realizations and observations are deactived between iterations.
  The function is testing reactivation as well. It is a bit tricky since there is no
  reactivation function. What is done is to start with identical copies, testdata and
  testdata2. In the first iteration, one observation is removed in testdata2 and before
  computing the update. In the subsequent iterations, testdata is used which includes
  the datapoint initially removed from testdata2.
*/

void test_deactivate_observations_and_realizations(const char *testdata_file) {
    res::es_testdata testdata(testdata_file);
    res::es_testdata testdata2(testdata_file);
    int num_iter = 10;
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);

    ies::data::Data ies_data(testdata.active_ens_size);
    ies::Config ies_config(true);

    Eigen::MatrixXd A0 = testdata.make_state("prior");
    Eigen::MatrixXd A = A0;

    ies_config.set_truncation(1.00);
    ies_config.max_steplength = 0.50;
    ies_config.min_steplength = 0.50;
    ies_config.inversion = ies::IES_INVERSION_SUBSPACE_EXACT_R;

    // deactivate an observation initially to test reactivation in the following iteration
    testdata2.deactivate_obs(2);

    ies::init_update(ies_data, testdata2.ens_mask, testdata2.obs_mask);
    ies::updateA(ies_data, A, testdata2.S, testdata2.R, testdata2.E,
                 testdata2.D, ies_config.inversion, ies_config.get_truncation(),
                 ies_config.get_steplength(ies_data.iteration_nr));
    ies_data.iteration_nr++;

    for (int iter = 1; iter < num_iter; iter++) {
        // Deactivate a realization
        if (iter == 3) {
            int iens = testdata.active_ens_size / 2;
            testdata.deactivate_realization(iens);
            A = Eigen::MatrixXd::Zero(
                A0.rows(), std::count(testdata.ens_mask.begin(),
                                      testdata.ens_mask.end(), true));
            int iens_active = 0;
            for (int iens = 0; iens < A0.cols(); iens++) {
                if (testdata.ens_mask[iens]) {
                    eigen_copy_column(A, A0, iens_active, iens);
                    iens_active += 1;
                }
            }
        }

        // Now deactivate a previously active observation
        if (iter == 7)
            testdata.deactivate_obs(testdata.active_obs_size / 2);

        ies::init_update(ies_data, testdata.ens_mask, testdata.obs_mask);
        ies::updateA(ies_data, A, testdata.S, testdata.R, testdata.E,
                     testdata.D, ies_config.inversion,
                     ies_config.get_truncation(),
                     ies_config.get_steplength(ies_data.iteration_nr));
        ies_data.iteration_nr++;
    }
    rng_free(rng);
}

int main(int argc, char **argv) {
    res::es_testdata testdata(argv[1]);
    cmp_std_ies(testdata);
    cmp_std_ies_delrel(testdata);
    test_deactivate_observations_and_realizations(argv[1]);
}
