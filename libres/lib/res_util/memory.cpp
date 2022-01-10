#include <memory>
#include <iostream>
#include <fstream>
#include <sys/resource.h>

namespace ert {
namespace utils {

/**
 * Internal utils
 *
 * get_file() exists to make testing simple. Note that the weak-attribute
 * is crucial to allow the method to be replaced in tests.
 */
[[gnu::weak]] std::shared_ptr<std::istream> get_file(const char *filename) {
    auto stream = std::make_shared<std::ifstream>(filename);
    return stream;
}

std::size_t parse_meminfo_linux(const char *file, const char *pattern) {
    auto meminfo = get_file(file);
    if (meminfo->good()) {
        std::string tp;
        std::size_t info;
        while (std::getline(*meminfo, tp)) {
            if (sscanf(tp.data(), pattern, &info) == 1)
                return info / 1000;
        }
    }
    // Returning 0 indicates that no memory-info was found
    return 0;
}

/**
 * NOTE: Linux only.
 *
 * Returns available memory (Mb) or 0 if the information is unavailable,
 * for example on the Mac-platform.
 *
 * For details and explanation of the returned value, see
 * the field MemFree in the section on /proc/meminfo in
 *
 *   https://www.kernel.org/doc/Documentation/filesystems/proc.txt
 */
std::size_t system_ram_free(void) {
    return parse_meminfo_linux("/proc/meminfo", "MemFree: %zd kB");
}

/**
 * NOTE: Linux only.
 *
 * Returns the current memory (Mb) used by the process or 0 if the,
 * information is unavailable, for example on the Mac-platform.
 *
 * For details and explanation of the returned value, see
 * the field VmSize in section on /proc/self/status in
 *
 *   https://www.kernel.org/doc/Documentation/filesystems/proc.txt
 */
std::size_t process_memory(void) {
    return parse_meminfo_linux("/proc/self/status", "VmSize: %zd kB");
}

/**
 * NOTE: Linux only.
 *
 * Returns the maximum memory (Mb) used by the process so far or 0
 * if the information is unavailable, for example on the Mac-platform.
 *
 * For details and explanation of the returned value, see
 * the field VmMax in the section on /proc/self/status in
 *
 *   https://www.kernel.org/doc/Documentation/filesystems/proc.txt
 */
std::size_t process_max_memory(void) {
    return parse_meminfo_linux("/proc/self/status", "VmPeak: %zd kB");
}

/**
 * NOTE: Linux only.
 *
 * Returns the maximum resident set size (RSS) of calling process in Mb,
 * i.e the max RSS so far in the lifetime of the process. Returns 0 if
 * this information is unavailable.
 *
 * For details of RSS see https://en.wikipedia.org/wiki/Resident_set_size
 * and the manpage for the getrusage() system-call.
 */
std::size_t process_max_rss(void) {
    struct rusage r_usage;
    long ret = getrusage(RUSAGE_SELF, &r_usage);
    if (ret == 0)
        return r_usage.ru_maxrss / 1000;

    return 0;
}
} // namespace utils
} // namespace ert
