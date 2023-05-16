#include "ert/python.hpp"
#include <ert/ecl/ecl_sum.hpp>
#include <ert/util/stringlist.hpp>
#include <string>
#include <vector>

std::vector<std::string>
ensemble_config_get_summary_key_list(const char *key,
                                     const ecl_sum_type *refcase) {
    std::vector<std::string> keylist;
    if (util_string_has_wildcard(key)) {
        if (refcase != nullptr) {
            stringlist_type *keys = stringlist_alloc_new();
            ecl_sum_select_matching_general_var_list(
                refcase, key,
                keys); /* expanding the wildcard notation with help of the refcase. */
            for (int k = 0; k < stringlist_get_size(keys); k++) {
                keylist.push_back((std::string)stringlist_iget(keys, k));
            }
            stringlist_free(keys);
        }
    } else {
        keylist.push_back(std::string(key));
    }

    return keylist;
}

ERT_CLIB_SUBMODULE("ensemble_config", m) {
    m.def("get_summary_key_list",
          [](const char *key, Cwrap<ecl_sum_type> refcase) {
              return ensemble_config_get_summary_key_list(key, refcase);
          });
}
