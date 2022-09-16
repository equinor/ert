#ifndef ERT_CONCURRENCY_HPP
#define ERT_CONCURRENCY_HPP

#include <condition_variable>
#include <mutex>

// Standard support for semaphores arrived in C++20, so make our own for now
// Idea and most of code is from
//  https://raymii.org/s/tutorials/Cpp_std_async_with_a_concurrency_limit.html
// and the class-name is intentionally left like the original :)
class Semafoor {
public:
    explicit Semafoor(size_t count) : count(count) {}
    void lock() {
        std::unique_lock<std::mutex> lock(mutex);
        condition_variable.wait(lock, [this] { return (count != 0); });
        --count;
    }
    void unlock() {
        std::unique_lock<std::mutex> lock(mutex);
        ++count;
        condition_variable.notify_one();
    }

private:
    std::mutex mutex;
    std::condition_variable condition_variable;
    size_t count;
};

#endif
