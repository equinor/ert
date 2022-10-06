#pragma once

#include <optional>
#include <vector>

using ActiveList = std::optional<std::vector<size_t>>;

namespace active_list {
class ActiveListIterable {
    const std::vector<size_t> *const m_data;
    const size_t m_total_size;

    class iterator {
        const std::vector<size_t> *const m_data;
        const size_t m_total_size;
        size_t m_pos;

    public:
        iterator(const std::vector<size_t> *const data, const size_t total_size,
                 size_t pos)
            : m_data(data), m_total_size(total_size), m_pos(pos) {}

        bool operator!=(const iterator &other) const {
            return m_pos != other.m_pos;
        }

        std::pair<size_t, size_t> operator*() const {
            return m_data != nullptr ? std::pair{m_pos, (*m_data)[m_pos]}
                                     : std::pair{m_pos, m_pos};
        }

        iterator &operator++() {
            ++m_pos;
            return *this;
        }
    };

public:
    ActiveListIterable(const std::vector<size_t> *const data,
                       const size_t total_size)
        : m_data(data), m_total_size(total_size) {}

    iterator begin() const { return {m_data, m_total_size, 0}; }

    iterator end() const {
        return {m_data, m_total_size,
                m_data != nullptr ? m_data->size() : m_total_size};
    }
};

inline ActiveListIterable with_total_size(const ActiveList &al,
                                          size_t total_size) {
    return {al.has_value() ? &(*al) : nullptr, total_size};
}

inline size_t size(const ActiveList &al, size_t total_size) {
    return al.has_value() ? al->size() : total_size;
}

} // namespace active_list
