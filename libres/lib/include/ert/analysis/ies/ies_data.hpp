#ifndef IES_DATA_H
#define IES_DATA_H

#include <optional>
#include <vector>

#include <ert/util/rng.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>

#include <ert/analysis/ies/ies_config.hpp>

namespace ies {
namespace data {

constexpr const char *ITER_KEY = "ITER";

class Data {
public:
    Data(int ens_size);
    ~Data();

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

    const matrix_type *getA0() const;
    const matrix_type *getW() const;
    matrix_type *getW();
    const matrix_type *getE() const;

    int ens_size() const;
    int obs_mask_size() const;
    int active_obs_count() const;
    int ens_mask_size() const;

    void store_initialE(const matrix_type *E0);
    void augment_initialE(const matrix_type *E0);
    void store_initialA(const matrix_type *A);

    matrix_type *alloc_activeE() const;
    matrix_type *alloc_activeW() const;
    matrix_type *alloc_activeA() const;

private:
    int m_ens_size;
    bool m_converged;
    int m_iteration_nr;
    matrix_type *
        W; // Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1)

    std::optional<int> m_state_size;
    std::vector<bool> m_ens_mask{};
    std::vector<bool> m_obs_mask0{};
    std::vector<bool> m_obs_mask{};
    matrix_type *A0 = nullptr; // Prior ensemble used in Ei=A0 Omega_i
    matrix_type *E =
        nullptr; // Prior ensemble of measurement perturations (should be the same for all iterations)
};

} // namespace data
} // namespace ies

#endif
