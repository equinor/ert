#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/row_scaling.hpp>
#include <ert/util/rng.h>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies_data.hpp>
#include <ert/analysis/update.hpp>

/**
 * @brief Test of analysis update using posterior properties described in ert-docs: https://ert.readthedocs.io/en/latest/theory/ensemble_based_methods.html
 *
 */

namespace analysis {
void run_analysis_update_without_rowscaling(
    const ies::config::Config &module_config, ies::data::Data &module_data,
    const std::vector<bool> &ens_mask, const std::vector<bool> &obs_mask,
    const Eigen::MatrixXd &S, const Eigen::MatrixXd &E,
    const Eigen::MatrixXd &D, const Eigen::MatrixXd &R, Eigen::MatrixXd &A);

void run_analysis_update_with_rowscaling(
    const ies::config::Config &module_config, ies::data::Data &module_data,
    const Eigen::MatrixXd &S, const Eigen::MatrixXd &E,
    const Eigen::MatrixXd &D, const Eigen::MatrixXd &R,
    std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
        &parameters);
} // namespace analysis
const double a_true = 1.0;
const double b_true = 5.0;

struct model {
    double a;
    double b;

    model(double a, double b) : a(a), b(b) {}

    model(rng_type *rng) {
        double a_std = 2.0;
        double b_std = 2.0;
        // Priors with bias
        double a_bias = 0.5 * a_std;
        double b_bias = -0.5 * b_std;
        this->a = enkf_util_rand_normal(a_true + a_bias, a_std, rng);
        this->b = enkf_util_rand_normal(b_true + b_bias, b_std, rng);
    }

    double eval(double x) const { return this->a * x + this->b; }

    int size() { return 2; }
};

SCENARIO("Running analysis update with and without row scaling on linear model",
         "[analysis]") {

    GIVEN("Fixed prior and measurements") {
        int ens_size = GENERATE(10, 100, 200);
        ies::data::Data module_data(ens_size);
        ies::config::Config config(false);

        auto rng = rng_alloc(MZRAN, INIT_DEFAULT);

        std::vector<bool> ens_mask(ens_size, true);
        auto meas_data = meas_data_alloc(ens_mask);
        auto obs_data = obs_data_alloc(1.0);

        model true_model{a_true, b_true};

        std::vector<model> ens;
        for (int iens = 0; iens < ens_size; iens++)
            ens.emplace_back(rng);

        // prior
        int nparam = true_model.size();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nparam, ens_size);
        for (int iens = 0; iens < ens_size; iens++) {
            const auto &model = ens[iens];
            A(0, iens) = model.a;
            A(1, iens) = model.b;
        }
        double a_avg_prior = A.row(0).sum() / ens_size;
        double b_avg_prior = A.row(1).sum() / ens_size;

        // observations and measurements
        int obs_size = 45;
        std::vector<double> sd_obs_values{10000.0, 100.0, 10.0, 1.0, 0.1};

        const char *obs_key = "OBS1";
        meas_block_type *mb =
            meas_data_add_block(meas_data, obs_key, 1, obs_size);
        obs_block_type *ob = obs_data_add_block(obs_data, obs_key, obs_size);
        std::vector<double> xarg(obs_size);
        for (int i = 0; i < obs_size; i++) {
            xarg[i] = i;
        }
        for (int iens = 0; iens < ens_size; iens++) {
            const auto &m = ens[iens];
            for (int iobs = 0; iobs < obs_size; iobs++)
                meas_block_iset(mb, iens, iobs, m.eval(xarg[iobs]));
        }
        std::vector<double> observations(obs_size);
        for (int iobs = 0; iobs < obs_size; iobs++) {
            // When observations != true model, then ml estimates != true parameters.
            // This gives both a more advanced and realistic test.
            // Standard normal N(0,1) noise is added to obtain this.
            // The randomness ensures we are not gaming the test.
            // But the difference could in principle be any non-zero scalar.
            observations[iobs] = true_model.eval(xarg[iobs]) +
                                 enkf_util_rand_normal(0.0, 1.0, rng);
        }

        // Leading to fixed Maximum likelihood estimate.
        // It will equal true values when observations are sampled without noise.
        // It will also stay the same over beliefs.
        double obs_mean = 0.0;
        double xarg_sum = 0.0;
        double xarg_sum_squared = 0.0;
        for (int iobs = 0; iobs < obs_size; iobs++) {
            obs_mean += observations[iobs];
            xarg_sum += xarg[iobs];
            xarg_sum_squared += std::pow(xarg[iobs], 2);
        }
        obs_mean /= obs_size;
        double iobs_mean = xarg_sum / obs_size;
        double a_ml_numerator = 0.0;
        for (int iobs = 0; iobs < obs_size; iobs++) {
            a_ml_numerator += xarg[iobs] * (observations[iobs] - obs_mean);
        }
        double a_ml =
            a_ml_numerator / (xarg_sum_squared - iobs_mean * xarg_sum);
        double b_ml = obs_mean - a_ml * iobs_mean;

        // Store posterior results when iterating over belief in observations
        int n_sd = sd_obs_values.size();
        std::vector<double> a_avg_posterior(n_sd);
        std::vector<double> b_avg_posterior(n_sd);
        std::vector<double> d_posterior_ml(n_sd);
        std::vector<double> d_prior_posterior(n_sd);

        WHEN("Iterating over belief in observations without row scaling") {
            for (int i_sd = 0; i_sd < n_sd; i_sd++) {
                double obs_std = sd_obs_values[i_sd];
                for (int iobs = 0; iobs < obs_size; iobs++) {
                    // The improtant part: observations stay the same
                    // What is iterated is the belief in them
                    obs_block_iset(ob, iobs, observations[iobs], obs_std);
                }

                int active_obs_size = obs_data_get_active_size(obs_data);
                Eigen::MatrixXd noise =
                    Eigen::MatrixXd::Zero(active_obs_size, ens_size);
                for (int j = 0; j < ens_size; j++)
                    for (int i = 0; i < active_obs_size; i++)
                        noise(i, j) = enkf_util_rand_normal(0, 1, rng);
                Eigen::VectorXd observation_values =
                    obs_data_values_as_vector(obs_data);
                Eigen::VectorXd observation_errors =
                    obs_data_errors_as_vector(obs_data);

                Eigen::MatrixXd S = meas_data_makeS(meas_data);
                Eigen::MatrixXd E = ies::makeE(observation_errors, noise);
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
                    observation_errors.rows(), observation_errors.rows());
                Eigen::MatrixXd D = ies::makeD(observation_values, E, S);

                Eigen::VectorXd error_inverse =
                    observation_errors.array().inverse();
                S = S.array().colwise() * error_inverse.array();
                E = E.array().colwise() * error_inverse.array();
                D = D.array().colwise() * error_inverse.array();

                const std::vector<bool> obs_mask =
                    obs_data_get_active_mask(obs_data);
                Eigen::MatrixXd A_iter = A; // Preserve prior
                analysis::run_analysis_update_without_rowscaling(
                    config, module_data, ens_mask, obs_mask, S, E, D, R,
                    A_iter);

                // Extract estimates
                a_avg_posterior[i_sd] = A_iter.row(0).sum() / ens_size;
                b_avg_posterior[i_sd] = A_iter.row(1).sum() / ens_size;

                // Calculate distances
                d_posterior_ml[i_sd] =
                    std::sqrt(std::pow((a_avg_posterior[i_sd] - a_ml), 2) +
                              std::pow((b_avg_posterior[i_sd] - b_ml), 2));
                d_prior_posterior[i_sd] = std::sqrt(
                    std::pow((a_avg_prior - a_avg_posterior[i_sd]), 2) +
                    std::pow((b_avg_prior - b_avg_posterior[i_sd]), 2));
            }

            // Test everything to some small (but generous) numeric precision
            double eps = 1e-2;

            // Compare with the prior-ml distance
            double d_prior_ml = std::sqrt(std::pow((a_avg_prior - a_ml), 2) +
                                          std::pow((b_avg_prior - b_ml), 2));

            THEN("All posterior estimates lie between prior and ml estimate") {
                for (int i_sd = 0; i_sd < n_sd; i_sd++) {
                    REQUIRE(d_posterior_ml[i_sd] - d_prior_ml < eps);
                    REQUIRE(d_prior_posterior[i_sd] - d_prior_ml < eps);
                }
            }

            THEN("Posterior parameter estimates improve with increased "
                 "trust in observations") {
                for (int i_sd = 1; i_sd < n_sd; i_sd++) {
                    REQUIRE(d_posterior_ml[i_sd] - d_posterior_ml[i_sd - 1] <
                            eps);
                }
            }

            THEN("At week beliefs, we should be close to the prior estimate") {
                REQUIRE(d_prior_posterior[0] < eps);
            }

            THEN("At strong beliefs, we should be close to the ml-estimate") {
                REQUIRE(d_posterior_ml[n_sd - 1] < eps);
            }
        }

        WHEN("Row scaling factor is 0 for both parameters") {
            auto row_scaling = std::make_shared<RowScaling>();
            for (int row = 0; row < nparam; row++)
                row_scaling->assign(row, 0.0);

            Eigen::MatrixXd A_with_scaling = A;

            std::vector parameters{std::pair{A_with_scaling, row_scaling}};

            for (int iobs = 0; iobs < obs_size; iobs++) {
                obs_block_iset(ob, iobs, observations[iobs], 1.0);
            }
            int active_obs_size = obs_data_get_active_size(obs_data);
            Eigen::MatrixXd noise =
                Eigen::MatrixXd::Zero(active_obs_size, ens_size);
            for (int j = 0; j < ens_size; j++)
                for (int i = 0; i < active_obs_size; i++)
                    noise(i, j) = enkf_util_rand_normal(0, 1, rng);
            Eigen::VectorXd observation_values =
                obs_data_values_as_vector(obs_data);
            Eigen::VectorXd observation_errors =
                obs_data_errors_as_vector(obs_data);
            Eigen::MatrixXd E = ies::makeE(observation_errors, noise);

            Eigen::MatrixXd S = meas_data_makeS(meas_data);
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
                observation_errors.rows(), observation_errors.rows());
            Eigen::MatrixXd D = ies::makeD(observation_values, E, S);

            Eigen::VectorXd error_inverse =
                observation_errors.array().inverse();
            S = S.array().colwise() * error_inverse.array();
            E = E.array().colwise() * error_inverse.array();
            D = D.array().colwise() * error_inverse.array();

            analysis::run_analysis_update_with_rowscaling(
                config, module_data, S, E, D, R, parameters);

            THEN("Updated parameter matrix should equal prior parameter "
                 "matrix") {
                REQUIRE(A == A_with_scaling);
            }
        }

        WHEN("Row scaling factor is 1 for both parameters") {
            auto row_scaling = std::make_shared<RowScaling>();
            for (int row = 0; row < nparam; row++)
                row_scaling->assign(row, 1.0);

            Eigen::MatrixXd A_no_scaling = A;

            std::vector parameters{std::pair{A, row_scaling}};

            for (int iobs = 0; iobs < obs_size; iobs++) {
                obs_block_iset(ob, iobs, observations[iobs], 1.0);
            }
            int active_obs_size = obs_data_get_active_size(obs_data);
            Eigen::MatrixXd noise =
                Eigen::MatrixXd::Zero(active_obs_size, ens_size);
            for (int j = 0; j < ens_size; j++)
                for (int i = 0; i < active_obs_size; i++)
                    noise(i, j) = enkf_util_rand_normal(0, 1, rng);
            Eigen::VectorXd observation_values =
                obs_data_values_as_vector(obs_data);
            Eigen::VectorXd observation_errors =
                obs_data_errors_as_vector(obs_data);
            Eigen::MatrixXd E = ies::makeE(observation_errors, noise);
            Eigen::MatrixXd S = meas_data_makeS(meas_data);
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
                observation_errors.rows(), observation_errors.rows());
            Eigen::MatrixXd D = ies::makeD(observation_values, E, S);

            Eigen::VectorXd error_inverse =
                observation_errors.array().inverse();
            S = S.array().colwise() * error_inverse.array();
            E = E.array().colwise() * error_inverse.array();
            D = D.array().colwise() * error_inverse.array();

            const std::vector<bool> obs_mask =
                obs_data_get_active_mask(obs_data);
            analysis::run_analysis_update_without_rowscaling(
                config, module_data, ens_mask, obs_mask, S, E, D, R,
                A_no_scaling);
            analysis::run_analysis_update_with_rowscaling(
                config, module_data, S, E, D, R, parameters);

            THEN("Updated parameter matrix with row scaling should equal "
                 "updated parameter matrix without row scaling") {
                auto A_with_scaling = parameters.begin()->first;
                REQUIRE(A_no_scaling.isApprox(A_with_scaling));
            }
        }

        WHEN("Row scaling factor is 0 for one parameter and 1 for the other") {
            auto row_scaling = std::make_shared<RowScaling>();
            row_scaling->assign(0, 1.0);
            row_scaling->assign(1, 0.0);

            Eigen::MatrixXd A_no_scaling = A;

            std::vector parameters{std::pair{A, row_scaling}};

            for (int iobs = 0; iobs < obs_size; iobs++) {
                obs_block_iset(ob, iobs, observations[iobs], 1.0);
            }
            int active_obs_size = obs_data_get_active_size(obs_data);
            Eigen::MatrixXd noise =
                Eigen::MatrixXd::Zero(active_obs_size, ens_size);
            for (int j = 0; j < ens_size; j++)
                for (int i = 0; i < active_obs_size; i++)
                    noise(i, j) = enkf_util_rand_normal(0, 1, rng);
            Eigen::VectorXd observation_values =
                obs_data_values_as_vector(obs_data);
            Eigen::VectorXd observation_errors =
                obs_data_errors_as_vector(obs_data);
            Eigen::MatrixXd E = ies::makeE(observation_errors, noise);

            Eigen::MatrixXd S = meas_data_makeS(meas_data);
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
                observation_errors.rows(), observation_errors.rows());
            Eigen::MatrixXd D = ies::makeD(observation_values, E, S);

            Eigen::VectorXd error_inverse =
                observation_errors.array().inverse();
            S = S.array().colwise() * error_inverse.array();
            E = E.array().colwise() * error_inverse.array();
            D = D.array().colwise() * error_inverse.array();

            const std::vector<bool> obs_mask =
                obs_data_get_active_mask(obs_data);
            analysis::run_analysis_update_with_rowscaling(
                config, module_data, S, E, D, R, parameters);
            analysis::run_analysis_update_without_rowscaling(
                config, module_data, ens_mask, obs_mask, S, E, D, R,
                A_no_scaling);

            THEN("First row of scaled parameters should equal first row of "
                 "unscaled parameters, while second row of scaled parameters "
                 "should equal prior") {
                auto A_with_scaling = parameters.begin()->first;
                Eigen::MatrixXd A_with_scaling_T = A_with_scaling.transpose();
                Eigen::MatrixXd A_no_scaling_T = A_no_scaling.transpose();
                Eigen::MatrixXd A_prior_T = A.transpose();
                REQUIRE(
                    A_with_scaling_T.col(0).isApprox(A_no_scaling_T.col(0)));
                REQUIRE(A_with_scaling_T.col(1).isApprox(A_prior_T.col(1)));
            }
        }

        WHEN("Iterating over belief in measurements with row scaling") {
            for (int i_sd = 0; i_sd < n_sd; i_sd++) {
                double obs_std = sd_obs_values[i_sd];
                for (int iobs = 0; iobs < obs_size; iobs++) {
                    // The improtant part: observations stay the same
                    // What is iterated is the belief in them
                    obs_block_iset(ob, iobs, observations[iobs], obs_std);
                }
                Eigen::MatrixXd A_with_scaling = A;

                auto row_scaling = std::make_shared<RowScaling>();
                row_scaling->assign(0, 1.0);
                row_scaling->assign(1, 0.7);

                std::vector parameters{std::pair{A_with_scaling, row_scaling}};
                int active_obs_size = obs_data_get_active_size(obs_data);
                Eigen::MatrixXd noise =
                    Eigen::MatrixXd::Zero(active_obs_size, ens_size);
                for (int j = 0; j < ens_size; j++)
                    for (int i = 0; i < active_obs_size; i++)
                        noise(i, j) = enkf_util_rand_normal(0, 1, rng);
                Eigen::VectorXd observation_values =
                    obs_data_values_as_vector(obs_data);
                Eigen::VectorXd observation_errors =
                    obs_data_errors_as_vector(obs_data);
                Eigen::MatrixXd E = ies::makeE(observation_errors, noise);

                Eigen::MatrixXd S = meas_data_makeS(meas_data);
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
                    observation_errors.rows(), observation_errors.rows());
                Eigen::MatrixXd D = ies::makeD(observation_values, E, S);

                Eigen::VectorXd error_inverse =
                    observation_errors.array().inverse();
                S = S.array().colwise() * error_inverse.array();
                E = E.array().colwise() * error_inverse.array();
                D = D.array().colwise() * error_inverse.array();
                const std::vector<bool> obs_mask =
                    obs_data_get_active_mask(obs_data);
                analysis::run_analysis_update_with_rowscaling(
                    config, module_data, S, E, D, R, parameters);

                // Extract estimates
                a_avg_posterior[i_sd] = A_with_scaling.row(0).sum() / ens_size;
                b_avg_posterior[i_sd] = A_with_scaling.row(1).sum() / ens_size;

                // Calculate distances
                d_posterior_ml[i_sd] =
                    std::sqrt(std::pow((a_avg_posterior[i_sd] - a_ml), 2) +
                              std::pow((b_avg_posterior[i_sd] - b_ml), 2));
                d_prior_posterior[i_sd] = std::sqrt(
                    std::pow((a_avg_prior - a_avg_posterior[i_sd]), 2) +
                    std::pow((b_avg_prior - b_avg_posterior[i_sd]), 2));
            }

            // Test everything to some small (but generous) numeric precision
            double eps = 1e-2;

            // Compare with the prior-ml distance
            double d_prior_ml = std::sqrt(std::pow((a_avg_prior - a_ml), 2) +
                                          std::pow((b_avg_prior - b_ml), 2));

            THEN("All posterior estimates lie between prior and ml estimate") {
                for (int i_sd = 0; i_sd < n_sd; i_sd++) {
                    REQUIRE(d_posterior_ml[i_sd] - d_prior_ml < eps);
                    REQUIRE(d_prior_posterior[i_sd] - d_prior_ml < eps);
                }
            }

            THEN("Posterior parameter estimates improve with increased "
                 "trust in observations") {
                for (int i_sd = 1; i_sd < n_sd; i_sd++) {
                    REQUIRE(d_posterior_ml[i_sd] - d_posterior_ml[i_sd - 1] <
                            eps);
                }
            }

            THEN("At week beliefs, we should be close to the prior estimate") {
                REQUIRE(d_prior_posterior[0] < eps);
            }
        }

        rng_free(rng);
        obs_data_free(obs_data);
        meas_data_free(meas_data);
    }
}
