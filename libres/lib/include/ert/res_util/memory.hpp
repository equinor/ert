#ifndef ERT_MEMORY_H
#define ERT_MEMORY_H

#include <cstddef>
#include <ert/logging.hpp>

namespace ert {
namespace utils {

std::size_t system_ram_free(void);
std::size_t process_memory(void);
std::size_t process_max_memory(void);
std::size_t process_max_rss(void);

class scoped_memory_logger {
public:
    scoped_memory_logger(std::shared_ptr<ert::ILogger> logger,
                         const std::string &message)
        : m_logger(logger), m_message(message), m_enter_mem(process_memory()),
          m_enter_max_mem(process_max_memory()) {

        if (m_enter_mem == 0 || m_enter_max_mem == 0)
            m_logger->info(
                "Enter {} Memory information not available on this platform",
                m_message);
        else
            m_logger->info("Enter {} Mem: {}Mb  MaxMem: {}Mb", m_message,
                           m_enter_mem, m_enter_max_mem);
    };

    ~scoped_memory_logger() {
        if (m_enter_mem == 0 || m_enter_max_mem == 0)
            m_logger->info(
                "Exit  {} Memory information not available on this platform",
                m_message);
        else {
            std::size_t cur_mem = process_memory();
            std::string delta_cur_mem =
                (m_enter_mem > cur_mem)
                    ? "-" + std::to_string(m_enter_mem - cur_mem)
                    : "+" + std::to_string(cur_mem - m_enter_mem);

            std::size_t max_mem = process_max_memory();
            std::string delta_max_mem =
                (m_enter_max_mem > max_mem)
                    ? "-" + std::to_string(m_enter_max_mem - max_mem)
                    : "+" + std::to_string(max_mem - m_enter_max_mem);

            m_logger->info("Exit  {} Mem: {}Mb ({}) MaxMem: {}Mb ({})",
                           m_message, cur_mem, delta_cur_mem, max_mem,
                           delta_max_mem);
        }
    };

private:
    std::shared_ptr<ert::ILogger> m_logger;
    const std::string m_message;
    std::size_t m_enter_mem, m_enter_max_mem;
};

} // namespace utils
} // namespace ert

#endif
