#include <string>
#include <filesystem>

#include "catch2/catch.hpp"

#include "ert/analysis/fwd_step_log.hpp"

namespace {
class TmpLog {
public:
    std::filesystem::path log_file_path;
    fwd_step_log_type *fwd_step_log;

    TmpLog() : log_file_path("tmp_fwd_step_log_open/test.txt") {
        fwd_step_log = fwd_step_log_alloc();
        fwd_step_log_set_log_file(fwd_step_log, log_file_path.c_str());
    }
    ~TmpLog() {
        fwd_step_log_free(fwd_step_log);
        std::filesystem::remove_all(log_file_path.parent_path());
    }
};
} // namespace

TEST_CASE_METHOD(TmpLog, "fwd_step_log_open clear_log is true", "[analysis]") {
    bool exists_before = std::filesystem::exists(log_file_path);
    REQUIRE(!exists_before);

    fwd_step_log_set_clear_log(fwd_step_log, true);
    fwd_step_log_open(fwd_step_log);

    bool exists_after = std::filesystem::exists(log_file_path);
    REQUIRE(exists_after);
}

TEST_CASE_METHOD(TmpLog, "fwd_step_log_open clear_log is false", "[analysis]") {
    bool exists_before = std::filesystem::exists(log_file_path);
    REQUIRE(!exists_before);

    fwd_step_log_set_clear_log(fwd_step_log, false);
    fwd_step_log_open(fwd_step_log);

    bool exists_after = std::filesystem::exists(log_file_path);
    REQUIRE(exists_after);
}
