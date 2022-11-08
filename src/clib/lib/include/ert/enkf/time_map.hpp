#pragma once

#include <ctime>
#include <filesystem>
#include <fmt/chrono.h>
#include <optional>
#include <vector>

typedef struct ecl_sum_struct ecl_sum_type;

struct Inconsistency {
    size_t step;
    std::time_t expected;
    std::time_t actual;
};

template <> struct fmt::formatter<Inconsistency> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Inconsistency &err, FormatContext &ctx) const
        -> decltype(ctx.out()) {
        std::tm expected_tm{};
        std::tm actual_tm{};
        gmtime_r(&err.expected, &expected_tm);
        gmtime_r(&err.actual, &actual_tm);
        return fmt::format_to(
            ctx.out(),
            "Time mismatch for step: {}, response time: "
            "{:04d}-{:02d}-{:02d}, reference case: {:04d}-{:02d}-{:02d}",
            err.step, expected_tm.tm_year + 1900, expected_tm.tm_mon + 1,
            expected_tm.tm_mday, actual_tm.tm_year + 1900, actual_tm.tm_mon + 1,
            actual_tm.tm_mday);
    }
};

class TimeMap : public std::vector<time_t> {
    using Super = std::vector<time_t>;
    const ecl_sum_type *m_refcase{};

    std::optional<Inconsistency> m_update(size_t step, time_t time);

public:
    using Super::vector;

    static constexpr time_t DEFAULT_TIME = -1;

    void write_binary(const std::filesystem::path &) const;
    void read_binary(const std::filesystem::path &);
    void read_text(const std::filesystem::path &);

    std::string summary_update(const ecl_sum_type *ecl_sum);
    bool attach_refcase(const ecl_sum_type *refcase);

    std::vector<int> indices(const ecl_sum_type *ecl_sum) const;
};
