#ifndef IES_CONFIG_H
#define IES_CONFIG_H

#include <variant>

namespace ies {

constexpr double DEFAULT_TRUNCATION = 0.98;
constexpr const char *IES_LOGFILE_KEY = "IES_LOGFILE";
constexpr const char *ENKF_SUBSPACE_DIMENSION_KEY = "ENKF_SUBSPACE_DIMENSION";
constexpr const char *IES_INVERSION_KEY = "IES_INVERSION";
constexpr const char *ENKF_TRUNCATION_KEY = "ENKF_TRUNCATION";
constexpr const char *IES_MAX_STEPLENGTH_KEY = "IES_MAX_STEPLENGTH";
constexpr const char *IES_MIN_STEPLENGTH_KEY = "IES_MIN_STEPLENGTH";
constexpr const char *IES_DEC_STEPLENGTH_KEY = "IES_DEC_STEPLENGTH";
constexpr const char *IES_DEBUG_KEY = "IES_DEBUG";
constexpr const char *ENKF_NCOMP_KEY = "ENKF_NCOMP";
constexpr const char *INVERSION_KEY = "INVERSION";
constexpr const char *STRING_INVERSION_EXACT = "EXACT";
constexpr const char *STRING_INVERSION_SUBSPACE_EXACT_R = "SUBSPACE_EXACT_R";
constexpr const char *STRING_INVERSION_SUBSPACE_EE_R = "SUBSPACE_EE_R";
constexpr const char *STRING_INVERSION_SUBSPACE_RE = "SUBSPACE_RE";

class Config {
public:
    explicit Config(bool ies_mode);

    void subspace_dimension(int subspace_dimension);
    void set_truncation(double truncation);
    const std::variant<double, int> &get_truncation() const;
    double get_dec_steplength() const;
    void set_dec_steplength(double dec_step);

    double get_steplength(int iteration_nr) const;

    /** Controlled by config key: DEFAULT_IES_INVERSION */
    int inversion;
    bool iterable;
    /** Controlled by config key: DEFAULT_IES_MAX_STEPLENGTH_KEY */
    double max_steplength;
    /** Controlled by config key: DEFAULT_IES_MIN_STEPLENGTH_KEY */
    double min_steplength;

private:
    /** Used for setting threshold of eigen values or number of eigen values */
    std::variant<double, int> m_truncation;
    /** Controlled by config key: DEFAULT_IES_DEC_STEPLENGTH_KEY */
    double m_dec_steplength;
};

} // namespace ies
#endif
