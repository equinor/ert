#pragma once

#include <cstddef>
#include <utility>

namespace ert {
template <typename IterT>
class Enumerator {
public:
    class Iterator;

private:
    IterT m_begin;
    IterT m_end;

public:
    Enumerator(IterT begin, IterT end):
        m_begin(begin), m_end(end) {}

    Iterator begin() { return {m_begin}; }
    Iterator end() { return {m_end}; }
};

template <typename IterT>
class Enumerator<IterT>::Iterator {
    IterT m_iter;
    size_t m_index{};
public:
    Iterator(IterT iter): m_iter(iter) {}

    bool operator==(const Iterator &other) const { return m_iter == other.m_iter; }

    Iterator& operator++() {
        ++m_iter;
        ++m_index;
        return *this;
    }

    std::pair<size_t, decltype(*m_iter)> operator*() {
        return {m_index, *m_iter};
    }
};

template <typename ContainerT>
auto enumerate(ContainerT &&container) -> Enumerator<typename ContainerT::iterator> {
    return {container.begin(), container.end()};
}

template <typename IterT>
auto enumerate(IterT begin, IterT end) -> Enumerator<IterT> {
    return {begin, end};
}
}
