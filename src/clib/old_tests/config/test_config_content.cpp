#include <ert/util/test_util.hpp>

#include <ert/config/config_content.hpp>

void test_create() {
    config_content_type *content = config_content_alloc("filename");
    config_content_free(content);
}

int main(int argc, char **argv) { test_create(); }
