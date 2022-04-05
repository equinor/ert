#include <vector>

namespace analysis {
std::vector<int>
bool_vector_to_active_list(const std::vector<bool> &bool_vector) {
    std::vector<int> active_list;
    for (int i = 0; i < bool_vector.size(); i++) {
        if (bool_vector[i])
            active_list.push_back(i);
    }
    return active_list;
}
} // namespace analysis
