#include <pybind11/embed.h>

[[gnu::constructor]]
static void initialize_interpreter() {
    pybind11::initialize_interpreter();
}

[[gnu::destructor]]
static void finalize_interpreter() {
    pybind11::finalize_interpreter();
}
