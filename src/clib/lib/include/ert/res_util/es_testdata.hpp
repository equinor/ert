#ifndef ES_TESTDATA_HPP
#define ES_TESTDATA_HPP

#include <Eigen/Dense>
#include <ert/util/bool_vector.hpp>
#include <string>
#include <vector>

namespace res {
class es_testdata {
public:
    std::string path;

    Eigen::MatrixXd S{};
    Eigen::MatrixXd E{};
    Eigen::MatrixXd R{};
    Eigen::MatrixXd D{};
    Eigen::MatrixXd dObs{};
    int active_obs_size{};
    int active_ens_size{};
    std::vector<bool> obs_mask;
    std::vector<bool> ens_mask;
    int state_size{};

    es_testdata(const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
                const Eigen::MatrixXd &D, const Eigen::MatrixXd &E,
                const Eigen::MatrixXd &dObs);
    es_testdata(const char *path);

    Eigen::MatrixXd make_matrix(const std::string &name, int rows,
                                int columns) const;
    void save_matrix(const std::string &name, const Eigen::MatrixXd &m) const;
    Eigen::MatrixXd make_state(const std::string &name) const;
    void save(const std::string &path) const;
    void deactivate_obs(int iobs);
    void deactivate_realization(int iens);
};

} // namespace res

#endif
