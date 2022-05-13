/*
   Copyright (C) 2019  Equinor ASA, Norway.

   The file 'ies_enkf.cpp' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/
#include <algorithm>
#include <variant>
#include <vector>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/python.hpp>

#include <ert/analysis/enkf_linalg.hpp>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies_data.hpp>

using Eigen::MatrixXd;

/**
 * @brief Implementation of algorithm as described in
 * "Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching"
 * https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
 *
 */
namespace ies {
void linalg_compute_AA_projection(const Eigen::MatrixXd &A, Eigen::MatrixXd &Y);

Eigen::MatrixXd linalg_solve_S(const Eigen::MatrixXd &W0,
                               const Eigen::MatrixXd &Y);

void linalg_subspace_inversion(Eigen::MatrixXd &W0, const int ies_inversion,
                               const Eigen::MatrixXd &E,
                               const Eigen::MatrixXd &R,
                               const Eigen::MatrixXd &S,
                               const Eigen::MatrixXd &H,
                               const std::variant<double, int> &truncation,
                               double ies_steplength);

void linalg_exact_inversion(Eigen::MatrixXd &W0, const int ies_inversion,
                            const Eigen::MatrixXd &S, const Eigen::MatrixXd &H,
                            double ies_steplength);
} // namespace ies

namespace {
auto logger = ert::get_logger("ies");
} // namespace

void ies::init_update(ies::data::Data &module_data,
                      const std::vector<bool> &ens_mask,
                      const std::vector<bool> &obs_mask) {
    module_data.update_ens_mask(ens_mask);
    module_data.store_initial_obs_mask(obs_mask);
    module_data.update_obs_mask(obs_mask);
}

Eigen::MatrixXd
ies::makeX(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Y0,
           const Eigen::MatrixXd &R, const Eigen::MatrixXd &E,
           const Eigen::MatrixXd &D, const ies::inversion_type ies_inversion,
           const std::variant<double, int> &truncation, Eigen::MatrixXd &W0,
           double ies_steplength, int iteration_nr)

{
    const int ens_size = Y0.cols();

    Eigen::MatrixXd Y = Y0;

    double nsc = 1.0 / sqrt(ens_size - 1.0);

    /*  Subtract mean of predictions to generate predicted ensemble anomaly matrix (Line 5) */
    Y = nsc * (Y.colwise() -
               Y.rowwise().mean()); // Y = Y * (I - (1 / ens_size) * 11')

    /* COMPUTING THE PROJECTION Y= Y * (Ai^+ * Ai) (only used when state_size < ens_size-1) */
    if (A.rows() > 0 && A.cols() > 0) {
        const int state_size = A.rows();
        if (state_size <= ens_size - 1) {
            ies::linalg_compute_AA_projection(A, Y);
        }
    }

    /*
     * When solving the system S = Y inv(Omega) we write
     *   Omega^T S^T = Y^T (line 6)
     */
    Eigen::MatrixXd S = ies::linalg_solve_S(W0, Y);

    /* INNOVATION H = S*W + D - Y   from Eq. (41) (Line 8)*/
    Eigen::MatrixXd H = D + S * W0; // H=D=dobs + E - Y

    /* Store previous W for convergence test */
    Eigen::MatrixXd W = W0;

    /*
     * COMPUTE NEW UPDATED W                                                                        (Line 9)
     *    W = W + ies_steplength * ( W - S'*(S*S'+R)^{-1} H )          (a)
     * which in the case when R=I can be rewritten as
     *    W = W + ies_steplength * ( W - (S'*S + I)^{-1} * S' * H )    (b)
     *
     * With R=I the subspace inversion (ies_inversion=1) solving Eq. (a) with singular value
     * trucation=1.000 gives exactly the same solution as the exact inversion (ies_inversion=0).
     *
     * Using ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(2), and a step length of 1.0,
     * one update gives identical result to STD as long as the same SVD
     * truncation is used.
     *
     * With very large data sets it is likely that the inversion becomes poorly
     * conditioned and a trucation=1.000 is not a good choice. In this case the
     * ies_inversion > 0 and truncation set to 0.99 or so, should stabelize
     * the algorithm.
     *
     * Using ies_inversion=IES_INVERSION_SUBSPACE_EE_R(3) and
     * ies_inversion=IES_INVERSION_SUBSPACE_RE(2) gives identical results but
     * ies_inversion=IES_INVERSION_SUBSPACE_RE is much faster (N^2m) than
     * ies_inversion=IES_INVERSION_SUBSPACE_EE_R (Nm^2).
     *
     * See the enum: ies_inverson in ies_config.hpp:
     *
     * ies_inversion=IES_INVERSION_EXACT(0)            -> exact inversion from (b) with exact R=I
     * ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(1) -> subspace inversion from (a) with exact R
     * ies_inversion=IES_INVERSION_SUBSPACE_EE_R(2)    -> subspace inversion from (a) with R=EE
     * ies_inversion=IES_INVERSION_SUBSPACE_RE(3)      -> subspace inversion from (a) with R represented by E
     */

    if (ies_inversion != ies::IES_INVERSION_EXACT) {
        ies::linalg_subspace_inversion(W0, ies_inversion, E, R, S, H,
                                       truncation, ies_steplength);
    } else if (ies_inversion == ies::IES_INVERSION_EXACT) {
        ies::linalg_exact_inversion(W0, ies_inversion, S, H, ies_steplength);
    }

    /*
     * CONSTRUCT TRANFORM MATRIX X FOR CURRENT ITERATION (Line 10)
     *   X= I + W/sqrt(N-1)
     */
    Eigen::MatrixXd X = W0;
    X /= sqrt(ens_size - 1.0);
    X.diagonal().array() += 1;

    /* COMPUTE ||W0 - W|| AND EVALUATE COST FUNCTION FOR PREVIOUS ITERATE (Line 12)*/
    Eigen::MatrixXd DW = W0 - W;

    std::vector<double> costJ(ens_size);
    double local_costf = 0.0;
    for (int i = 0; i < ens_size; i++) {
        costJ[i] = W.col(i).dot(W.col(i)) + D.col(i).dot(D.col(i));
        local_costf += costJ[i];
    }
    local_costf = local_costf / ens_size;

    logger->info("IES  iter:{} cost function: {}", iteration_nr, local_costf);
    return X;
}

void ies::updateA(data::Data &data,
                  // Updated ensemble A retured to ERT.
                  Eigen::Ref<Eigen::MatrixXd> A,
                  // Ensemble of predicted measurements
                  const Eigen::MatrixXd &Yin,
                  // Measurement error covariance matrix (not used)
                  const Eigen::MatrixXd &Rin,
                  // Ensemble of observation perturbations
                  const Eigen::MatrixXd &Ein,
                  // (d+E-Y) Ensemble of perturbed observations - Y
                  const Eigen::MatrixXd &Din,
                  const ies::inversion_type ies_inversion,
                  const std::variant<double, int> &truncation,
                  double ies_steplength) {

    // Number of active realizations in current iteration
    int ens_size = Yin.cols();
    int state_size = A.rows();

    int iteration_nr = data.iteration_nr();

    data.update_state_size(state_size);
    /*
      Counting number of active observations for current iteration. If the
      observations have been used in previous iterations they are contained in
      data->E0. If they are introduced in the current iteration they will be
      augmented to data->E.
    */
    data.store_initialE(Ein);
    data.augment_initialE(Ein);
    data.store_initialA(A);

    /*
     * Re-structure input matrices according to new active obs_mask and ens_size.
     * Allocates the local matrices to be used.
     * Copies the initial measurement perturbations for the active observations into the current E matrix.
     * Copies the inputs in D, Y and R into their local representations
     */
    Eigen::MatrixXd E = data.make_activeE();
    Eigen::MatrixXd D = Din;

    /* Subtract new measurement perturbations              D=D-E    */
    D -= Ein;
    /* Add old measurement perturbations */
    D += E;

    double costf;

    auto W0 = data.make_activeW();
    Eigen::MatrixXd X;

    X = makeX(A, Yin, Rin, E, D, ies_inversion, truncation, W0, ies_steplength,
              iteration_nr);

    ies::linalg_store_active_W(&data, W0);

    /* COMPUTE NEW ENSEMBLE SOLUTION FOR CURRENT ITERATION  Ei=A0*X (Line 11)*/
    Eigen::MatrixXd A0 = data.make_activeA();
    A = A0 * X;
}

/*  COMPUTING THE PROJECTION Y= Y * (Ai^+ * Ai) (only used when state_size < ens_size-1)    */
void ies::linalg_compute_AA_projection(const Eigen::MatrixXd &A,
                                       Eigen::MatrixXd &Y) {

    Eigen::MatrixXd Ai = A;
    Ai = Ai.colwise() - Ai.rowwise().mean();
    auto svd = Ai.bdcSvd(Eigen::ComputeThinV);
    Eigen::MatrixXd VT = svd.matrixV().transpose();
    Eigen::MatrixXd AAi = VT.transpose() * VT;
    Y *= AAi;
}

/*
* COMPUTE  Omega= I + W (I-11'/sqrt(ens_size))    from Eq. (36).                                   (Line 6)
*  When solving the system S = Y inv(Omega) we write
*     Omega^T S^T = Y^T
*/
Eigen::MatrixXd ies::linalg_solve_S(const Eigen::MatrixXd &W0,
                                    const Eigen::MatrixXd &Y) {

    /*  Here we compute the W (I-11'/N) / sqrt(N-1)  and transpose it).*/
    Eigen::MatrixXd Omega =
        W0; // Omega=data->W (from previous iteration used to solve for S)
    double nsc = 1.0 / sqrt(W0.cols() - 1.0);
    Omega =
        nsc * (Omega.colwise() -
               Omega.rowwise().mean()); // Omega = Omega * (I - (1 / N) * 11')
    Omega.transposeInPlace();           // Omega=transpose(Omega)
    Omega.diagonal().array() += 1.0;

    Eigen::MatrixXd ST = Omega.fullPivLu().solve(Y.transpose());

    return ST.transpose();
}

/*
*  The standard inversion works on the equation
*          S'*(S*S'+R)^{-1} H           (a)
*/
void ies::linalg_subspace_inversion(
    Eigen::MatrixXd &W0, const int ies_inversion, const Eigen::MatrixXd &E,
    const Eigen::MatrixXd &R, const Eigen::MatrixXd &S,
    const Eigen::MatrixXd &H, const std::variant<double, int> &truncation,
    double ies_steplength) {

    int ens_size = S.cols();
    int nrobs = S.rows();
    double nsc = 1.0 / sqrt(ens_size - 1.0);
    Eigen::MatrixXd X1 = Eigen::MatrixXd::Zero(
        nrobs, std::min(ens_size, nrobs)); // Used in subspace inversion
    Eigen::VectorXd eig(ens_size);

    if (ies_inversion == IES_INVERSION_SUBSPACE_RE) {
        Eigen::MatrixXd scaledE = E;
        scaledE *= nsc;
        enkf_linalg_lowrankE(S, scaledE, X1, eig, truncation);

    } else if (ies_inversion == IES_INVERSION_SUBSPACE_EE_R) {
        Eigen::MatrixXd Et = E.transpose();
        MatrixXd Cee = E * Et;
        Cee *= 1.0 / ((ens_size - 1) * (ens_size - 1));

        enkf_linalg_lowrankCinv(S, Cee, X1, eig, truncation);

    } else if (ies_inversion == IES_INVERSION_SUBSPACE_EXACT_R) {
        Eigen::MatrixXd scaledR = R;
        scaledR *= nsc * nsc;
        enkf_linalg_lowrankCinv(S, scaledR, X1, eig, truncation);
    }

    /*
        X3 = X1 * diag(eig) * X1' * H (Similar to Eq. 14.31, Evensen (2007))
    */
    Eigen::Map<Eigen::VectorXd> eig_vector(eig.data(), eig.size());
    Eigen::MatrixXd X3 = enkf_linalg_genX3(X1, H, eig_vector);

    /*    Update data->W = (1-ies_steplength) * data->W +  ies_steplength * S' * X3                          (Line 9)    */
    W0 = ies_steplength * S.transpose() * X3 + (1.0 - ies_steplength) * W0;
}

/*
*  The standard inversion works on the equation
*          S'*(S*S'+R)^{-1} H           (a)
*  which in the case when R=I can be rewritten as
*          (S'*S + I)^{-1} * S' * H     (b)
*/
void ies::linalg_exact_inversion(Eigen::MatrixXd &W0, const int ies_inversion,
                                 const Eigen::MatrixXd &S,
                                 const Eigen::MatrixXd &H,
                                 double ies_steplength) {
    int ens_size = S.cols();

    MatrixXd StS = MatrixXd::Identity(ens_size, ens_size) + S.transpose() * S;

    auto svd = StS.bdcSvd(Eigen::ComputeFullU);
    MatrixXd Z = svd.matrixU();
    Eigen::VectorXd eig = svd.singularValues();

    MatrixXd ZtStH = Z.transpose() * S.transpose() * H;

    for (int i = 0; i < ens_size; i++)
        ZtStH.row(i) /= eig[i];

    /*    Update data->W = (1-ies_steplength) * data->W +  ies_steplength * Z * (Lamda^{-1}) Z' S' H         (Line 9)    */
    W0 = ies_steplength * Z * ZtStH + (1.0 - ies_steplength) * W0;
}

/*
* the updated W is stored for each iteration in data->W. If we have lost realizations we copy only the active rows and cols from
* W0 to data->W which is then used in the algorithm.  (note the definition of the pointer dataW to data->W)
*/
void ies::linalg_store_active_W(ies::data::Data *data,
                                const Eigen::MatrixXd &W0) {
    int ens_size_msk = data->ens_mask_size();
    int i = 0;
    int j;
    Eigen::MatrixXd &dataW = data->getW();
    const std::vector<bool> &ens_mask = data->ens_mask();
    dataW.setConstant(0.0);
    for (int iens = 0; iens < ens_size_msk; iens++) {
        if (ens_mask[iens]) {
            j = 0;
            for (int jens = 0; jens < ens_size_msk; jens++) {
                if (ens_mask[jens]) {
                    dataW(iens, jens) = W0(i, j);
                    j += 1;
                }
            }
            i += 1;
        }
    }
}

Eigen::MatrixXd ies::makeE(const Eigen::VectorXd &obs_errors,
                           const Eigen::MatrixXd &noise) {
    int active_obs_size = obs_errors.rows();
    int active_ens_size = noise.cols();

    Eigen::MatrixXd E = noise;
    Eigen::VectorXd pert_mean = E.rowwise().mean();
    E = E.colwise() - pert_mean;
    Eigen::VectorXd pert_var = E.cwiseProduct(E).rowwise().sum();

    for (int i = 0; i < active_obs_size; i++) {
        double factor = obs_errors(i) * sqrt(active_ens_size / pert_var(i));
        E.row(i) *= factor;
    }

    return E;
}

Eigen::MatrixXd ies::makeD(const Eigen::VectorXd &obs_values,
                           const Eigen::MatrixXd &E, const Eigen::MatrixXd &S) {

    Eigen::MatrixXd D = E - S;

    for (int i = 0; i < obs_values.rows(); i++)
        for (int iens = 0; iens < D.cols(); iens++)
            D(i, iens) += obs_values(i);

    return D;
}

RES_LIB_SUBMODULE("ies", m) {
    m.def("make_X", ies::makeX, py::arg("A"), py::arg("Y0"), py::arg("R"),
          py::arg("E"), py::arg("D"), py::arg("ies_inversion"),
          py::arg("truncation"), py::arg("W0"), py::arg("ies_steplength"),
          py::arg("iteration_nr"));
    m.def("make_E", ies::makeE, py::arg("obs_errors"), py::arg("noise"));
    m.def("make_D", ies::makeD, py::arg("obs_values"), py::arg("E"),
          py::arg("S"));
    m.def("update_A", ies::updateA, py::arg("data"), py::arg("A"),
          py::arg("Yin"), py::arg("R"), py::arg("E"), py::arg("D"),
          py::arg("inversion"), py::arg("truncation"), py::arg("step_length"));
    m.def("init_update", ies::init_update, py::arg("module_data"),
          py::arg("ens_mask"), py::arg("obs_mask"));
}
