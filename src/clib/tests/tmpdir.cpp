#include <optional>
#include <regex>
#include <string>

#include <pwd.h>
#include <unistd.h>

#include <catch2/catch.hpp>
#include <ert/except.hpp>

#include "tmpdir.hpp"

using namespace std::string_literals;
namespace fs = std::filesystem;

namespace {
/**
 * Convert string into a nicer directory name
 *
 * Downcases, replaces whitespace-ish chars with '_' and removes the
 * rest.
 */
std::string clean_prefix(const std::string &prefix) {
    std::string out;
    out.reserve(prefix.size());

    for (char c : prefix) {
        if ('a' <= c && c <= 'z')
            out.push_back(c);
        else if ('A' <= c && c <= 'Z')
            out.push_back(c + 'a' - 'A');
        else if (c == '_' || c == '-' || c == ' ')
            out.push_back('_');
    }
    return out;
}

/**
 * Get POSIX username of the current user
 *
 * Falls back to the user's UID if their username is unavailable.
 */
std::string get_username() {
    auto uid = getuid();
    auto pw = getpwuid(uid);
    return pw ? std::string{pw->pw_name} : std::to_string(uid);
}

/**
 * Create a new directory suffixed with a number
 *
 * The prefix is normalised with 'clean_string'.
 *
 * \param basepath Directory in which to create the new directory
 * \param prefix Prefix of the directory to be created
 */
std::filesystem::path make_numbered_dir(const fs::path &basepath,
                                        const std::string &prefix) {
    auto norm_prefix = clean_prefix(prefix);
    std::regex re{norm_prefix + "(\\d+)"s};
    std::smatch match;

    // Find largest entry with the given 'prefix'
    int number = 0;
    for (auto entry : fs::directory_iterator(basepath)) {
        auto entry_str = entry.path().filename().string();
        if (!std::regex_match(entry_str, match, re))
            continue;
        number = std::max(std::stoi(match[1]) + 1, number);
    }

    // A race condition might occur between us finding the largest suffix number
    // and us creating a directory. Eg, by running two test suites
    // simulatenously. To avoid this, we try the next 10 suffix numbers. Should
    // `fs::create_directory` succeed and return true, we know that we were the
    // ones who created the directory and have exclusive access to it.
    for (int i{}; i < 10; ++i) {
        std::error_code ec;
        auto path = basepath / (norm_prefix + std::to_string(number + i));
        if (fs::create_directory(path, ec)) {
            // Create a symlink. Error codes are ignored because we don't really care
            auto current = basepath / (norm_prefix + "current"s);
            fs::remove(current, ec);
            fs::create_directory_symlink(path, current, ec);
            return path;
        }
    }

    throw exc::runtime_error("Could not make numbered dir in {} with prefix {}",
                             basepath.string(), norm_prefix);
}

/**
 * Get the base tmpdir for this entire test session
 */
const fs::path &get_basedir() {
    static std::optional<fs::path> basedir;
    if (basedir.has_value())
        return *basedir;

    auto tmp = fs::temp_directory_path();
    tmp /= fmt::format("catch2-of-{}", get_username());

    auto name = Catch::getCurrentContext().getConfig()->name();
    fs::create_directories(tmp);
    basedir.emplace(make_numbered_dir(tmp, name));

    // Create a symlink. Error codes are ignored because we don't really care
    auto current = tmp / (clean_prefix(name) + "current"s);
    std::error_code ec;
    fs::remove(current, ec);
    fs::create_directory_symlink(*basedir, current, ec);

    return *basedir;
}

fs::path make_path_for_catch2_test() {
    return make_numbered_dir(get_basedir(),
                             Catch::getResultCapture().getCurrentTestName());
}

} // namespace

TmpDir::TmpDir() : m_prev_cwd(fs::current_path()) {
    auto tmpdir = make_path_for_catch2_test();
    UNSCOPED_INFO("Using temporary directory " << tmpdir.string() << "\n");
    fs::current_path(tmpdir);
}

TmpDir::~TmpDir() { fs::current_path(m_prev_cwd); }

/**
 * Get the current temporary directory for this object
 * @return the path of the current temporary directory as std::string
 */
std::string TmpDir::get_current_tmpdir() { return fs::current_path().string(); }

TEST_CASE("Create a single tmpdir", "[tmpdir]") {
    auto before = fs::current_path();
    {
        WITH_TMPDIR;
        REQUIRE(fs::current_path() != before);
    }
    REQUIRE(fs::current_path() == before);
}

TEST_CASE("Create multiple tmpdirs", "[tmpdir]") {
    const int N = 10;
    fs::path paths[N];

    auto before = fs::current_path();
    for (int i{}; i < N; ++i) {
        WITH_TMPDIR;
        paths[i] = fs::current_path();
    }

    for (int i{}; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            REQUIRE(paths[i] != paths[j]);
        }
    }
    REQUIRE(fs::current_path() == before);
}
