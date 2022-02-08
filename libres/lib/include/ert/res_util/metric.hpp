#include <chrono>
#include <memory>

#include <ert/logging.hpp>

namespace ert {
namespace utils {
class Benchmark {
public:
    Benchmark(std::shared_ptr<ILogger> logger, const std::string &message)
        : m_logger(logger)
        , m_message(message) {}

    ~Benchmark() {
        m_logger->info(
            "{}'s execution time: {:.4f} seconds", m_message,
            std::chrono::duration<float>(clock::now() - m_start).count());
    }

private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point m_start = clock::now();
    const std::string m_message;
    std::shared_ptr<ILogger> m_logger;
};

} // namespace utils
} // namespace ert
