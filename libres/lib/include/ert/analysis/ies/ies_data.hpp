#ifndef IES_DATA_H
#define IES_DATA_H

#include <optional>
#include <vector>
#include <Eigen/Dense>

namespace ies {
namespace data {

constexpr const char *ITER_KEY = "ITER";

class Data {
public:
    Data(int ens_size);

    void iteration_nr(int iteration_nr);
    int iteration_nr() const;
    int inc_iteration_nr();

    void update_ens_mask(const std::vector<bool> &mask);
    void store_initial_obs_mask(const std::vector<bool> &mask);
    void update_obs_mask(const std::vector<bool> &mask);
    void update_state_size(int state_size);

    const std::vector<bool> &obs_mask0() const;
    const std::vector<bool> &obs_mask() const;
    const std::vector<bool> &ens_mask() const;

    const Eigen::MatrixXd &getA0() const;
    const Eigen::MatrixXd &getW() const;
    Eigen::MatrixXd &getW();
    const Eigen::MatrixXd &getE() const;

    int ens_size() const;
    int obs_mask_size() const;
    int active_obs_count() const;
    int ens_mask_size() const;

    void store_initialE(const Eigen::MatrixXd &E0);
    void augment_initialE(const Eigen::MatrixXd &E0);
    void store_initialA(const Eigen::MatrixXd &A);

    Eigen::MatrixXd make_activeE() const;
    Eigen::MatrixXd make_activeW() const;
    Eigen::MatrixXd make_activeA() const;

private:
    int m_ens_size;
    bool m_converged;
    int m_iteration_nr;
    Eigen::MatrixXd
        W; // Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1)

    std::optional<int> m_state_size;
    std::vector<bool> m_ens_mask{};
    std::vector<bool> m_obs_mask0{};
    std::vector<bool> m_obs_mask{};
    Eigen::MatrixXd A0{}; // Prior ensemble used in Ei=A0 Omega_i
    Eigen::MatrixXd
        E; // Prior ensemble of measurement perturations (should be the same for all iterations)
};

} // namespace data
} // namespace ies

#endif
